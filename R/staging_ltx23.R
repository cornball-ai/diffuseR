#' Pinned Staging for Phase-Sequential Components
#'
#' Phase offloading moves each large component (transformer,
#' connectors, VAEs, vocoder) between CPU and GPU every render. From
#' pageable memory those copies run through the driver's bounce
#' buffer at a fraction of PCIe speed; page-locked (pinned) host
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
#' and \code{options(diffuseR.pin_staging = FALSE)} before
#' \code{ltx23_load_pipeline} opts out (e.g. under host memory
#' pressure, where unswappable pages turn thrashing into OOM).
#'
#' @name staging_ltx23
NULL

#' Pin a component's tensors for fast phase transfer
#'
#' @param module An nn_module on the CPU.
#'
#' @return A list of \code{list(live, pinned)} tensor pairs, or NULL
#'   if pinning is unavailable (no CUDA, or page-locking failed).
#'
#' @keywords internal
# Allocate pinned host memory without Tensor$pin_memory(): this torch
# build's binding requires the deprecated device argument (omitting it
# errors with "Expected a torch_device"; passing it prints two libtorch
# deprecation warnings per tensor - thousands of lines per pipeline
# load). torch_empty_strided is the one creation op that exposes
# pin_memory, so pin by allocating and copying; fall back to the noisy
# path on builds where that fails.
.ltx23_pin_host <- function(p) {
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

.ltx23_pin_component <- function(module) {
    if (!torch::cuda_is_available()) {
        return(NULL)
    }
    tryCatch({
        tensors <- c(module$parameters, module$buffers)
        lapply(tensors, function(p) {
            pinned <- .ltx23_pin_host(p)
            p$set_data(pinned)
            list(live = p, pinned = pinned)
        })
    }, error = function(e) NULL)
}

#' Move a pinned component onto the compute device
#'
#' Non-blocking copies from pinned memory share the default stream,
#' so later kernels are ordered after them; no explicit sync needed.
#'
#' @keywords internal
.ltx23_staged_onload <- function(staging, device) {
    for (pair in staging) {
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
.ltx23_staged_offload <- function(staging) {
    for (pair in staging) {
        pair$live$set_data(pair$pinned)
    }
    invisible(NULL)
}
