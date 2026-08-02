# In-process model residency: pinned host weights, disposable GPU copies.
#
# A resident pipeline keeps its canonical weights as page-locked (pinned)
# CPU tensors for the life of the handle. Activation creates the GPU
# representation with a DMA copy from pinned memory; deactivation
# destroys only the GPU representation and re-points the modules at the
# pinned host storage. Reactivation never touches the disk, so handing
# a small GPU between models is a sub-second operation instead of a full
# pipeline reload.
#
# Same contract as whisper's and chatterbox's R/resident.R (the three
# packages present one interface to a residency broker), adapted to
# diffuseR's shape: a pipeline is a classed list holding SEVERAL
# nn_modules (transformer, decoder, text encoder(s), and for LTX a video
# VAE, audio VAE and vocoder), so components are discovered by scanning
# the pipeline for nn_module fields and the staging set is keyed by
# field name. Non-tensor fields (config, tokenizer, scheduler) ride
# along untouched.
#
# This layer sits ABOVE the per-generation phase offloading already in
# txt2img_flux()/txt2img_flux2()/txt2img_zimage()/txt2vid_ltx23(). Those
# swap one component at a time WITHIN a render; residency is about who
# owns the GPU BETWEEN renders. The two compose: an active handle renders
# with its normal internal phase behaviour, and deactivation releases
# whatever is still resident so a sibling model can take the card.
#
# Mechanics rest on two torch behaviours (verified in whisper's port,
# re-verified by this package's tests):
# - nn_module$to() REBINDS parameter/buffer objects, so pinned host
#   tensors held in res$staging survive activation, and any tensor handle
#   taken before a transition is stale after it. All re-binding therefore
#   resolves the modules' CURRENT tensors by name, every time.
# - Tensor$set_data() works across devices: a CUDA parameter can be
#   re-pointed directly at a pinned CPU tensor. That is the evict
#   mechanism; the orphaned CUDA storage is reclaimed by gc() +
#   cuda_empty_cache().
#
# States: inactive -> activating -> active -> deactivating -> inactive.
# Failed transitions roll back to pinned host state; a rollback that
# cannot be verified leaves the handle "broken" (fail-closed: only status
# and unload work). "unloaded" is terminal.

# Families that ship a pinned/staged loader. Keyed by the `model` name
# used everywhere else in the package (see recommend()).
.resident_families <- c("flux1", "flux2", "zimage", "ltx")

#' Every nn_module field of a pipeline, by name
#'
#' Discovery beats a hard-coded list: the families disagree on which
#' components exist (FLUX.1 has two text encoders, LTX adds a video VAE,
#' an audio VAE and a vocoder), and a field added later is picked up
#' without touching this file.
#'
#' @param pipeline A loaded diffuseR pipeline.
#'
#' @return A named list of the pipeline's \code{nn_module} fields,
#'   possibly empty.
#'
#' @keywords internal
.resident_components <- function(pipeline) {
    keep <- vapply(pipeline, function(x) inherits(x, "nn_module"), logical(1))
    pipeline[keep]
}

#' Pin every component of a pipeline for fast transfer
#'
#' Re-uses any staging the loader already built (the phase-offload path
#' pins as part of loading), and pins the rest. Pinning a component that
#' is currently on the GPU also evicts it, since \code{.pin_component}
#' copies into page-locked host memory and re-points the live tensors at
#' it, so this doubles as the initial offload.
#'
#' @param pipeline A loaded diffuseR pipeline.
#' @param verbose Print progress.
#'
#' @return A named list of staging sets, one per component that could be
#'   pinned. Components that fail to page-lock are absent, and fall back
#'   to the pageable \code{$to()} path.
#'
#' @keywords internal
.resident_pin <- function(pipeline, verbose = TRUE) {
    existing <- pipeline$staging %||% list()
    comps <- .resident_components(pipeline)
    if (verbose && length(comps)) {
        message("Pinning ", length(comps), " components for residency...")
    }
    staging <- list()
    for (nm in names(comps)) {
        if (!is.null(existing[[nm]])) {
            staging[[nm]] <- existing[[nm]]
            next
        }
        extra <- if (identical(nm, "transformer") &&
            isTRUE(pipeline$fp8_resident)) {
            .flux_fp8_collect(comps[[nm]])
        } else {
            NULL
        }
        st <- .pin_component(comps[[nm]], extra = extra)
        if (!is.null(st)) {
            staging[[nm]] <- st
        }
    }
    staging
}

#' Total pinned host bytes across a staging set
#'
#' @param staging A named list of staging sets.
#'
#' @return Numeric. Bytes of page-locked host memory held.
#'
#' @keywords internal
.resident_pinned_bytes <- function(staging) {
    total <- 0
    for (st in staging) {
        for (pair in st) {
            total <- total + prod(as.numeric(pair$pinned$shape)) *
            .dtype_bytes(pair$pinned$dtype)
        }
    }
    total
}

# Bytes per element, keyed by the libtorch dtype name that
# as.character() on a torch_dtype returns ("Float", "Half", "Byte",
# "Long", ...), NOT the R constructor alias. Unknown dtypes fall back to
# 4, which only affects a reported number.
.dtype_widths <- c(double = 8, long = 8, complexfloat = 8,
                   float = 4, int = 4,
                   half = 2, bfloat16 = 2, short = 2,
                   byte = 1, char = 1, bool = 1,
                   float8_e4m3fn = 1, float8_e5m2 = 1)

.dtype_bytes <- function(dtype) {
    nm <- tolower(tryCatch(as.character(dtype), error = function(e) ""))
    w <- .dtype_widths[[nm, exact = TRUE]]
    if (is.null(w)) 4 else w
}

#' TRUE when every staged tensor sits on the expected device type
#'
#' @param staging A named list of staging sets.
#' @param type "cpu" or "cuda".
#'
#' @return Logical.
#'
#' @keywords internal
.resident_all_on <- function(staging, type) {
    for (st in staging) {
        for (pair in st) {
            dev <- tryCatch(pair$live$device$type, error = function(e) NA_character_)
            if (!identical(dev, type)) {
                return(FALSE)
            }
        }
    }
    TRUE
}

#' Refuse operations that the current state cannot serve
#'
#' @param res A resident handle.
#' @param verb What the caller is attempting, for the message.
#'
#' @return Invisibly TRUE, or an error.
#'
#' @keywords internal
.resident_guard <- function(res, verb) {
    if (identical(res$state, "unloaded")) {
        stop("cannot ", verb, ": this handle is unloaded", call. = FALSE)
    }
    if (identical(res$state, "broken")) {
        stop("cannot ", verb, ": this handle is broken (", res$last_error %||%
                 "no detail recorded", "). Only resident_status() and ",
             "resident_unload() work from here.", call. = FALSE)
    }
    invisible(TRUE)
}

#' Load a diffusion pipeline as a resident handle
#'
#' Loads a pipeline once and keeps its weights page-locked on the host
#' for the life of the handle. The GPU representation is created by
#' \code{\link{resident_activate}} and destroyed by
#' \code{\link{resident_deactivate}}, so a 16 GB card can hand itself
#' between models without either one re-reading its weights from disk.
#'
#' The handle is bound to one explicit GPU at load: a bare \code{"cuda"}
#' resolves to the current device now, and every later transition uses
#' that index, so the handle cannot drift to whichever GPU happens to be
#' current at transition time.
#'
#' One caveat on multi-GPU hosts: the family loader itself runs on the
#' \emph{current} device, and only the residency handle is bound to
#' \code{device}. Loading with \code{device = "cuda:1"} from a session
#' whose current device is 0 therefore stages through GPU 0 before the
#' first activation lands on GPU 1. Wrap the call in
#' \code{torch::with_device(device = "cuda:1", ...)} when that matters.
#'
#' The pipeline is left \emph{inactive} (weights pinned on the host, no
#' VRAM held). Call \code{\link{resident_activate}} before generating.
#'
#' @param model One of "flux1", "flux2", "zimage", "ltx".
#' @param device Target CUDA device, e.g. "cuda" or "cuda:1".
#' @param ... Passed to the family loader (\code{\link{flux_load_pipeline}},
#'   \code{\link{flux2_load_pipeline}}, \code{\link{zimage_load_pipeline}}
#'   or \code{\link{ltx23_load_pipeline}}). \code{ltx} requires
#'   \code{checkpoint_path}.
#' @param verbose Print progress messages.
#'
#' @return A \code{diffuseR_resident} handle (an environment). Inspect it
#'   with \code{\link{resident_status}}; the fields of interest are the
#'   state, the bound device, the component names, and the pinned host
#'   byte count.
#'
#' @seealso \code{\link{resident_activate}}, \code{\link{resident_status}}
#'
#' @examples
#' \dontrun{
#' res <- resident_load("flux2")
#' resident_activate(res)
#' img <- resident_generate(res, "a cat in a spacesuit", seed = 7)
#' resident_deactivate(res) # VRAM freed, weights stay pinned in RAM
#' resident_activate(res) # fast: DMA copy, no disk
#' resident_unload(res)
#' }
#'
#' @export
resident_load <- function(model = c("flux2", "flux1", "zimage", "ltx"),
                          device = "cuda", ..., verbose = TRUE) {
    model <- match.arg(model)
    if (!torch::cuda_is_available()) {
        stop("resident_load() requires CUDA", call. = FALSE)
    }
    if (!grepl("^cuda", device)) {
        stop("resident_load() requires a CUDA device, got '", device, "'",
             call. = FALSE)
    }
    # Bind to one explicit GPU now, so later transitions cannot drift.
    bound <- if (identical(device, "cuda")) {
        paste0("cuda:", torch::cuda_current_device())
    } else {
        device
    }

    loader <- switch(model,
                     flux1 = flux_load_pipeline,
                     flux2 = flux2_load_pipeline,
                     zimage = zimage_load_pipeline,
                     ltx = ltx23_load_pipeline)
    pipeline <- loader(device = "cuda", verbose = verbose, ...)

    staging <- .resident_pin(pipeline, verbose = verbose)
    # Pinning also evicted anything the loader had left resident, so the
    # handle starts inactive with no VRAM held.
    .resident_release_vram()

    res <- new.env(parent = emptyenv())
    res$model <- model
    res$device <- bound
    res$pipeline <- pipeline
    res$staging <- staging
    res$components <- names(.resident_components(pipeline))
    res$pinned_bytes <- .resident_pinned_bytes(staging)
    res$state <- "inactive"
    res$last_error <- NULL
    res$loaded_at <- Sys.time()
    structure(res, class = "diffuseR_resident")
}

# gc() then empty the caching allocator. Split out so every transition
# releases VRAM the same way.
.resident_release_vram <- function() {
    gc()
    tryCatch(torch::cuda_empty_cache(), error = function(e) NULL)
    invisible(NULL)
}

#' Bring a resident pipeline onto the GPU
#'
#' Copies every pinned component to the handle's bound device by DMA and
#' verifies the result tensor-by-tensor. A failure rolls back to the
#' pinned host state; a rollback that cannot itself be verified leaves
#' the handle broken.
#'
#' @param res A \code{diffuseR_resident} handle.
#'
#' @return Invisibly the handle, with state "active".
#'
#' @export
resident_activate <- function(res) {
    stopifnot(inherits(res, "diffuseR_resident"))
    .resident_guard(res, "activate")
    if (identical(res$state, "active")) {
        return(invisible(res))
    }
    if (!identical(res$state, "inactive")) {
        stop("cannot activate from state '", res$state, "'", call. = FALSE)
    }
    res$state <- "activating"
    ok <- tryCatch({
        for (nm in names(res$staging)) {
            .staged_onload(res$staging[[nm]], res$device)
        }
        TRUE
    }, error = function(e) {
        res$last_error <- conditionMessage(e)
        FALSE
    })
    if (ok) {
        res$state <- "active"
        return(invisible(res))
    }
    # Roll back to pinned host state and verify it.
    rolled <- tryCatch({
        for (nm in names(res$staging)) {
            .staged_offload(res$staging[[nm]])
        }
        .resident_release_vram()
        .resident_all_on(res$staging, "cpu")
    }, error = function(e) FALSE)
    if (isTRUE(rolled)) {
        res$state <- "inactive"
        stop("resident_activate() failed (rolled back to pinned host ",
             "state): ", res$last_error, call. = FALSE)
    }
    res$state <- "broken"
    stop("resident_activate() failed and the rollback could not be ",
         "verified: ", res$last_error, call. = FALSE)
}

#' Release a resident pipeline's VRAM
#'
#' Re-points every component at its pinned host copy and drops the GPU
#' storage. Weights are immutable during inference, so the pinned copies
#' are still current and this moves no bytes: it is a pointer swap plus a
#' cache release. The handle stays loaded and can be reactivated without
#' touching the disk.
#'
#' @param res A \code{diffuseR_resident} handle.
#' @param release Empty the CUDA caching allocator afterwards. Leave TRUE
#'   unless another handle on the same device is about to reuse the pool.
#'
#' @return Invisibly the handle, with state "inactive".
#'
#' @export
resident_deactivate <- function(res, release = TRUE) {
    stopifnot(inherits(res, "diffuseR_resident"))
    .resident_guard(res, "deactivate")
    if (identical(res$state, "inactive")) {
        return(invisible(res))
    }
    if (!identical(res$state, "active")) {
        stop("cannot deactivate from state '", res$state, "'", call. = FALSE)
    }
    res$state <- "deactivating"
    verified <- tryCatch({
        for (nm in names(res$staging)) {
            .staged_offload(res$staging[[nm]])
        }
        if (isTRUE(release)) {
            .resident_release_vram()
        }
        .resident_all_on(res$staging, "cpu")
    }, error = function(e) {
        res$last_error <- conditionMessage(e)
        FALSE
    })
    if (isTRUE(verified)) {
        res$state <- "inactive"
        return(invisible(res))
    }
    res$state <- "broken"
    stop("resident_deactivate() could not verify the pinned host state; ",
         "the handle is broken and holds no usable GPU copy. ",
         res$last_error %||% "", call. = FALSE)
}

#' Generate from an active resident pipeline
#'
#' Dispatches to the family's generator with the resident pipeline
#' supplied, so no weights are re-read. The handle must be active.
#'
#' @param res A \code{diffuseR_resident} handle.
#' @param prompt Character. The text prompt.
#' @param ... Passed to \code{\link{txt2img_flux}},
#'   \code{\link{txt2img_flux2}}, \code{\link{txt2img_zimage}} or
#'   \code{\link{txt2vid_ltx2}}.
#'
#' @return Whatever the family generator returns: an image array for the
#'   image families, a video array for \code{ltx}.
#'
#' @export
resident_generate <- function(res, prompt, ...) {
    stopifnot(inherits(res, "diffuseR_resident"))
    .resident_guard(res, "generate")
    if (!identical(res$state, "active")) {
        stop("cannot generate from state '", res$state,
             "'; call resident_activate() first", call. = FALSE)
    }
    gen <- switch(res$model,
                  flux1 = txt2img_flux,
                  flux2 = txt2img_flux2,
                  zimage = txt2img_zimage,
                  ltx = txt2vid_ltx2)
    gen(prompt, pipeline = res$pipeline, ...)
}

#' Status of a resident handle
#'
#' @param res A \code{diffuseR_resident} handle.
#'
#' @return A list with \code{model}, \code{state}, \code{device},
#'   \code{components} (character vector of pinned component names),
#'   \code{pinned_bytes} (page-locked host bytes held),
#'   \code{gpu_allocated} and \code{gpu_reserved} (bytes the CUDA
#'   caching allocator reports live and held for this process, NA
#'   without CUDA), \code{loaded_at}, and \code{last_error} (NULL unless
#'   a transition failed).
#'
#' @export
resident_status <- function(res) {
    stopifnot(inherits(res, "diffuseR_resident"))
    mem <- .cuda_bytes()
    list(model = res$model,
         state = res$state,
         device = res$device,
         components = names(res$staging),
         pinned_bytes = res$pinned_bytes,
         gpu_allocated = mem$allocated,
         gpu_reserved = mem$reserved,
         loaded_at = res$loaded_at,
         last_error = res$last_error)
}

# Live and reserved CUDA bytes for this process. torch has no
# cuda_memory_allocated(); the numbers live under cuda_memory_stats(),
# which itself errors without a CUDA build, hence the tryCatch.
.cuda_bytes <- function() {
    s <- tryCatch(torch::cuda_memory_stats(), error = function(e) NULL)
    if (is.null(s)) {
        return(list(allocated = NA_real_, reserved = NA_real_))
    }
    list(allocated = s$allocated_bytes$all$current %||% NA_real_,
         reserved = s$reserved_bytes$all$current %||% NA_real_)
}

#' Drop a resident handle entirely
#'
#' Releases the GPU copy if any, drops the pipeline and the pinned host
#' storage, and marks the handle unloaded. Terminal: nothing but
#' \code{\link{resident_status}} works afterwards.
#'
#' @param res A \code{diffuseR_resident} handle.
#'
#' @return Invisibly the handle, with state "unloaded".
#'
#' @export
resident_unload <- function(res) {
    stopifnot(inherits(res, "diffuseR_resident"))
    if (identical(res$state, "unloaded")) {
        return(invisible(res))
    }
    # Best effort: a broken handle still gets its memory back.
    tryCatch({
        for (nm in names(res$staging)) {
            .staged_offload(res$staging[[nm]])
        }
    }, error = function(e) NULL)
    res$pipeline <- NULL
    res$staging <- list()
    res$components <- character(0)
    res$pinned_bytes <- 0
    res$state <- "unloaded"
    .resident_release_vram()
    invisible(res)
}

#' Print a resident handle
#'
#' @param x A \code{diffuseR_resident} handle.
#' @param ... Ignored.
#'
#' @return Invisibly \code{x}. Called for the side effect of printing a
#'   one-block summary to the console.
#'
#' @export
print.diffuseR_resident <- function(x, ...) {
    s <- resident_status(x)
    cat("<diffuseR resident>\n")
    cat("  model:      ", s$model, "\n", sep = "")
    cat("  state:      ", s$state, "\n", sep = "")
    cat("  device:     ", s$device, "\n", sep = "")
    cat("  components: ",
        if (length(s$components)) paste(s$components, collapse = ", ") else "-",
        "\n", sep = "")
    cat("  pinned:     ", .fmt_gb(s$pinned_bytes), "\n", sep = "")
    if (!is.na(s$gpu_allocated)) {
        cat("  gpu:        ", .fmt_gb(s$gpu_allocated), " allocated, ",
            .fmt_gb(s$gpu_reserved), " reserved\n", sep = "")
    }
    if (!is.null(s$last_error)) {
        cat("  last error: ", s$last_error, "\n", sep = "")
    }
    invisible(x)
}

# Byte count as GB, for the print method and messages.
.fmt_gb <- function(b) {
    if (is.null(b) || is.na(b) || b <= 0) {
        return("0 GB")
    }
    sprintf("%.2f GB", b / 1024^3)
}
