#' Pinned Staging for Phase-Sequential Components
#'
#' Phase offloading moves each large component (transformer,
#' connectors, VAEs, vocoder, text encoders) between CPU and GPU every
#' render. From pageable memory those copies run through the driver's
#' bounce buffer at a fraction of PCIe speed; page-locked (pinned) host
#' memory transfers by DMA at full rate. Each component's parameters
#' and buffers are pinned once at load; onload swaps every tensor to
#' a non-blocking GPU copy of its pinned source, and offload simply
#' re-points the tensors at the still-valid pinned copies — weights
#' are immutable during inference, so offload moves no bytes at all.
#'
#' Costs: the model's host copies become non-swappable for the life
#' of the pipeline (no extra RAM - set_data repoints the same
#' tensors), and page-locking adds ~9s to pipeline load. Measured
#' post byte-LUT (768x512x49, NF4, RTX 5060 Ti): ~7s saved per render
#' (warm renders 64-66s pageable vs 57-59s pinned; denoise and decode
#' identical, the delta is pure transfer), so pinning breaks even on
#' the second render and costs a single-render session ~2s net. On by
#' default; page-locking failure falls back silently per component,
#' and \code{options(diffuseR.pin_staging = FALSE)} before the loader
#' opts out (e.g. under host memory pressure, where unswappable pages
#' turn thrashing into OOM). The LTX pipeline, the Gemma3 encoder, and
#' the FLUX-family image loaders (flux1, flux2, zimage) all stage
#' pinned weights; \code{\link{recommend}} computes the RAM-aware
#' \code{pin} default per model.
#'
#' @name staging
#' @aliases staging_ltx23
NULL

# Allocate pinned host memory without Tensor$pin_memory(): this torch
# build's binding requires the deprecated device argument (omitting it
# errors with "Expected a torch_device"; passing it prints two libtorch
# deprecation warnings per tensor - thousands of lines per pipeline
# load). torch_empty_strided is the one creation op that exposes
# pin_memory, so pin by allocating and copying; fall back to the noisy
# path on builds where that fails.
.pin_host <- function(p) {
    tryCatch({
        sz <- as.integer(p$shape)
        st <- if (length(sz)) {
            as.integer(rev(cumprod(c(1, rev(sz)[-length(sz)]))))
        } else {
            integer(0)
        }
        buf <- torch::torch_empty_strided(sz, st, dtype = p$dtype,
            pin_memory = TRUE)
        torch::with_no_grad(buf$copy_(p))
        buf
    }, error = function(e) {
        suppressWarnings(p$pin_memory(device = torch::torch_device("cuda")))
    })
}

#' Pin a component's tensors for fast phase transfer
#'
#' @param module An nn_module on the CPU.
#' @param extra Optional list of additional plain-field tensors to pin
#'   alongside the module's parameters and buffers (e.g. an fp8
#'   linear's \code{weight_fp8}/\code{weight_scale} fields, which live
#'   outside \code{parameters}/\code{buffers}). \code{set_data} mutates
#'   each tensor in place, so the field reference stays valid.
#'
#' @return A list of \code{list(live, pinned)} tensor pairs, or NULL
#'   if pinning is unavailable (no CUDA, or page-locking failed).
#'
#' @keywords internal
.pin_component <- function(module, extra = NULL) {
    if (!torch::cuda_is_available()) {
        return(NULL)
    }
    tryCatch({
        tensors <- c(module$parameters, module$buffers, extra)
        lapply(tensors, function(p) {
            pinned <- .pin_host(p)
            p$set_data(pinned)
            list(live = p, pinned = pinned)
        })
    }, error = function(e) NULL)
}

# What a device spec names, as type and index. The index is NA when the
# spec leaves it open: "cuda" means whichever card is current, so a
# target without an index accepts any card, and "cuda:1" accepts only
# that one. A tensor's own device always carries a concrete index on the
# card (torch reports 0 for "cuda"), so the comparison below is exact
# whenever the caller asked for a particular card. Parsed from the string
# rather than through torch_device() so `.staged_on` runs without torch,
# which is how it is tested.
.device_spec <- function(device) {
    if (inherits(device, "torch_device")) {
        return(list(type = device$type,
                    index = as.integer(device$index %||% NA_integer_)))
    }
    s <- as.character(device)
    index <- if (grepl(":", s, fixed = TRUE)) {
        as.integer(sub("^[^:]*:", "", s))
    } else {
        NA_integer_
    }
    list(type = sub(":.*$", "", s), index = index)
}

# Is this tensor on the device the spec names? Type must match; the
# index must match too when the spec has one.
.on_device <- function(tensor, spec) {
    d <- tryCatch(tensor$device, error = function(e) NULL)
    if (is.null(d) || !identical(d$type, spec$type)) {
        return(FALSE)
    }
    if (is.na(spec$index)) {
        return(TRUE)
    }
    identical(as.integer(d$index %||% NA_integer_), spec$index)
}

#' Is every pinned tensor of a component on this device?
#'
#' The check a caller makes before skipping an onload. It asks EVERY
#' pair, not the first one: a component is on the card when all of it
#' is, and a probe of one tensor cannot tell a resident component from
#' one whose onload failed partway. That partial state is real -- an
#' onload that runs out of device memory leaves the pairs it copied on
#' the card and the rest on the host -- and a first-pair probe reports
#' it as "already resident", so every later phase skips the onload and
#' dies on a device mismatch, on every call, until the process ends.
#' That is how a gpuhost's LTX entry wedged for a whole show on
#' 2026-09-10: one failed encoder onload, then "mat2 is on cpu" from
#' every request after it.
#'
#' @param staging A component's staging set: the list of
#'   \code{list(live, pinned)} pairs \code{.pin_component} returned.
#' @param device The compute device, as a string (\code{"cuda"},
#'   \code{"cuda:1"}) or a \code{torch_device}. A spec without an index
#'   accepts any card; one with an index accepts only that card.
#' @return TRUE when every pair's live tensor is on that device; FALSE
#'   on any mismatch or unreadable pair. Vacuously TRUE for an empty
#'   staging set, which holds nothing to move.
#' @keywords internal
.staged_on <- function(staging, device) {
    spec <- .device_spec(device)
    for (pair in staging) {
        if (!.on_device(pair$live, spec)) {
            return(FALSE)
        }
    }
    TRUE
}

#' Move a pinned component onto the compute device
#'
#' Non-blocking copies from pinned memory share the default stream,
#' so later kernels are ordered after them; no explicit sync needed.
#'
#' Idempotent PER PAIR: a tensor already on the device is left where it
#' is, so a resident component costs nothing to onload again (no
#' re-transfer of weights over themselves, which fragments the
#' allocator pool) and a component whose earlier onload stopped partway
#' is completed rather than restarted.
#'
#' @keywords internal
.staged_onload <- function(staging, device) {
    spec <- .device_spec(device)
    for (pair in staging) {
        if (.on_device(pair$live, spec)) {
            next
        }
        pair$live$set_data(pair$pinned$to(device = device, non_blocking = TRUE))
    }
    invisible(NULL)
}

#' Return a pinned component to the CPU
#'
#' Weights are immutable during inference, so the pinned host copies
#' are still current: offload is a pointer swap, no transfer.
#'
#' @keywords internal
.staged_offload <- function(staging) {
    for (pair in staging) {
        pair$live$set_data(pair$pinned)
    }
    invisible(NULL)
}
