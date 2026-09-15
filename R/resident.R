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
.resident_families <- c("flux1", "flux2", "zimage", "ltx", "sdxl", "sd21")

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
.dtype_widths <- c(double = 8, long = 8, complexfloat = 8, float = 4,
                   int = 4, half = 2, bfloat16 = 2, short = 2, byte = 1,
                   char = 1, bool = 1, float8_e4m3fn = 1, float8_e5m2 = 1)

.dtype_bytes <- function(dtype) {
    nm <- tolower(tryCatch(as.character(dtype), error = function(e) ""))
    w <- .dtype_widths[[nm, exact = TRUE]]
    if (is.null(w)) {
        4
    } else {
        w
    }
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
            dev <- tryCatch(pair$live$device$type,
                            error = function(e) NA_character_)
            if (!identical(dev, type)) {
                return(FALSE)
            }
        }
    }
    TRUE
}

#' How many components actually have their tensors on the GPU
#'
#' Ground truth, as opposed to the handle's declared state. The two can
#' disagree: a pipeline built with \code{phase_offload = TRUE} swaps each
#' component back to pinned host memory as its phase finishes, so after a
#' render the handle is still "active" while the card holds nothing. A
#' broker deciding who to evict needs the measurement, not the claim.
#'
#' @param staging A named list of staging sets.
#'
#' @return Integer. Number of components whose live tensors are on CUDA.
#'
#' @keywords internal
.resident_on_gpu_count <- function(staging) {
    sum(vapply(staging, function(st) {
        isTRUE(length(st) > 0) &&
        identical(tryCatch(st[[1]]$live$device$type,
                           error = function(e) NA_character_), "cuda")
    }, logical(1)))
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
        stop("cannot ", verb, ": this handle is broken (",
             res$last_error %||% "no detail recorded",
             "). Only resident_status() and ",
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
#' @param model One of "flux1", "flux2", "zimage", "ltx", "sdxl".
#' @param device Target CUDA device, e.g. "cuda" or "cuda:1".
#' @param ... Passed to the family loader (\code{\link{flux_load_pipeline}},
#'   \code{\link{flux2_load_pipeline}}, \code{\link{zimage_load_pipeline}},
#'   \code{\link{ltx23_load_pipeline}} or
#'   \code{\link{sdxl_load_pipeline}}). \code{ltx} requires
#'   \code{checkpoint_path}; \code{sdxl} needs nothing (it defaults to the
#'   \code{\link{download_sdxl}} cache).
#' @param verbose Print progress messages.
#' @param text_encoder,tokenizer \code{ltx} only: paths to the Gemma3 encoder
#'   artifact and the tokenizer directory. \code{\link{ltx23_load_pipeline}}
#'   does not load these -- \code{\link{txt2vid_ltx2}} takes them per call --
#'   so a handle built without them can be activated and cannot generate.
#'   Given here they are loaded ONCE, pinned on the host, and passed to every
#'   generate; the encoder rides to the card for its phase and back off, the
#'   way the pipeline's own components do, so it is never resident beside the
#'   transformer.
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
resident_load <- function(model = c("flux2", "flux1", "zimage", "ltx",
                                    "sdxl", "sd21"),
                          device = "cuda", ..., verbose = TRUE,
                          text_encoder = NULL, tokenizer = NULL) {
    model <- match.arg(model)
    ## NAMED ARGUMENTS RATHER THAN `...`, because `...` goes to the family
    ## loader and `ltx23_load_pipeline` has no `...` of its own -- an unknown
    ## argument there is an error, not a pass-through. Refused for the other
    ## families for the same reason: silently ignoring them would leave a
    ## caller believing a text encoder had been loaded.
    if (!identical(model, "ltx") &&
        (!is.null(text_encoder) || !is.null(tokenizer))) {
        stop("text_encoder/tokenizer apply to the ltx family only; ",
             model, " loads its own", call. = FALSE)
    }
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
                     ltx = ltx23_load_pipeline,
                     sdxl = sdxl_load_pipeline,
                     sd21 = sd21_load_pipeline)
    # Capture the phase-offload choice here rather than reading it back
    # off the pipeline: the FLUX family stores it as a field, LTX takes
    # it again at generate time and stores nothing, so the field is
    # absent there and a NULL would be misread as "stays resident".
    # Every family loader defaults it TRUE.
    dots <- list(...)
    phase_offload <- if (is.null(dots$phase_offload)) {
        TRUE
    } else {
        isTRUE(dots$phase_offload)
    }
    pipeline <- loader(device = "cuda", verbose = verbose, ...)

    staging <- .resident_pin(pipeline, verbose = verbose)
    # Pinning also evicted anything the loader had left resident, so the
    # handle starts inactive with no VRAM held.
    .resident_release_vram()

    res <- new.env(parent = emptyenv())
    res$model <- model
    res$device <- bound
    res$pipeline <- pipeline
    res$phase_offload <- phase_offload
    res$staging <- staging
    # A family may want only part of its set on the card (see
    # .resident_gpu_set). NULL means all of it.
    res$gpu_components <- pipeline$gpu_components
    res$components <- names(.resident_components(pipeline))
    res$pinned_bytes <- .resident_pinned_bytes(staging)

    ## THE LTX TEXT ENCODER, LOADED ONCE AND KEPT OFF THE HANDLE'S STAGING.
    ##
    ## `txt2vid_ltx2` takes `text_encoder` and `tokenizer` per call and
    ## accepts a PATH, which it then loads -- so a serving caller that passed
    ## paths would re-read 7.6 GB of Gemma3 on every request. Loading here
    ## makes it once.
    ##
    ## Deliberately NOT added to `staging`: `resident_activate` puts
    ## everything in staging on the card at once, and the encoder beside the
    ## transformer does not fit. It carries its own staging attribute from
    ## `pin = TRUE`, and `encode_with_gemma3` onloads it for the encode and
    ## offloads on exit -- one GPU tenant per phase, the same discipline the
    ## pipeline's own components follow.
    if (identical(model, "ltx") && !is.null(text_encoder)) {
        if (is.null(tokenizer)) {
            stop("text_encoder needs a tokenizer: the encode takes both",
                 call. = FALSE)
        }
        if (verbose) message("Loading the Gemma3 text encoder (pinned)...")
        res$text_encoder <- load_gemma3_text_encoder(
            text_encoder, device = "cpu", pin = TRUE, verbose = verbose)
        res$tokenizer <- gemma3_tokenizer(tokenizer)
        ## Counted, so `resident_status()` reports what the process actually
        ## holds. A pinned set omitted from the total reads as headroom that
        ## is not there, and the fleet's admission arithmetic is downstream
        ## of this number.
        ## WRAPPED IN A LIST, AND THAT IS NOT COSMETIC. There are two
        ## staging shapes in this package: a pipeline's is a list OF
        ## COMPONENTS each holding a list of pairs, which is why
        ## `.resident_pinned_bytes` loops twice; an encoder's
        ## `attr(model, "staging")` is a FLAT list of pairs, which is why
        ## `.staged_onload` loops once. Passing the flat one straight in
        ## reads a pair's fields as pairs and dies on `pair$pinned$shape`
        ## -- "object of type 'closure' is not subsettable", from inside a
        ## worker, sixty seconds after the pin began.
        te_staging <- attr(res$text_encoder, "staging")
        if (!is.null(te_staging)) {
            res$pinned_bytes <- res$pinned_bytes +
                .resident_pinned_bytes(list(te_staging))
        }
    }
    res$state <- "inactive"
    res$last_error <- NULL
    res$loaded_at <- Sys.time()
    structure(res, class = "diffuseR_resident")
}

#' Grow the caching allocator's pool in one allocation before a bulk onload
#'
#' A cold bulk onload grows the pool one \code{cudaMalloc} per tensor, and
#' the syscalls dominate: SDXL's 8.0 GB pinned set measured 24.16 s on the
#' first activation against 0.32 s on the second, a 74x ratio, on an
#' otherwise idle RTX 5060 Ti. One large allocation freed straight back into
#' the pool lets the transfers carve from cached blocks instead. Same
#' technique the NF4 LTX loader already uses for its first transformer
#' onload (83.5 s -> 4.6 s there).
#'
#' It matters beyond the wall clock: a residency broker with a startup
#' deadline reads 24 s inside a first activate as a wedged worker.
#'
#' Best-effort. A card that cannot seat the block in one piece falls back to
#' the per-tensor path, which is the current behaviour and merely slow, so
#' the failure is swallowed rather than raised.
#'
#' Growing the pool is best-effort in the partial case. A request smaller
#' than a free block already in the cache is served from that block and
#' grows nothing, so when the pool is short by less than it already holds
#' the pre-warm may be absorbed rather than add capacity. That is bounded
#' and harmless -- the onload then falls back to the per-tensor path for
#' the remainder, which is the old behaviour -- and the alternative, asking
#' for the whole figure to force a new segment, is the accumulation bug
#' this function exists to avoid. The cold pool, which is the case worth
#' optimising and the one a broker's first request hits, is unaffected.
#'
#' @param bytes Numeric. Host bytes about to be transferred; the pool is
#'   warmed toward this plus a small margin for allocator slack.
#' @param device Target CUDA device, e.g. "cuda" or "cuda:1". Also selects
#'   which device's allocator is measured.
#' @param held Free cached bytes the allocator already holds on that
#'   device, i.e. reserved minus allocated -- bytes that are reserved but
#'   live belong to something else and cannot serve this transfer. NULL
#'   measures it. Pass a value to make the decision deterministic: without
#'   CUDA the measurement is 0, which would always warm, so a test that
#'   wants the skip has to state what the pool holds rather than depend on
#'   the machine having a card. Same reason
#'   \code{\link{.resident_check_fits}} takes \code{free_gb}.
#'
#' @return Invisibly, the bytes requested from the allocator: 0 when the
#'   pool already covers the transfer and nothing was asked for.
#'
#' @keywords internal
.resident_prewarm <- function(bytes, device, held = NULL) {
    # isTRUE() rather than a bare is.finite(): a NULL pinned_bytes gives
    # logical(0), and `||` on a zero-length value is an error in R >= 4.3,
    # so the guard meant to skip the pre-warm would instead fail the
    # activation it exists to speed up.
    if (!isTRUE(is.finite(bytes)) || bytes <= 0) {
        return(invisible(0))
    }
    # Only grow what is missing, and only when something IS missing.
    #
    # Asking for the full figure unconditionally doubles the pool on every
    # activation after the first. A render fragments the cache into its
    # activation blocks, so the next single large request cannot be served
    # from it and takes a fresh cudaMalloc alongside the old block. With
    # release = FALSE -- which a residency broker passes deliberately, to
    # keep an exclusive grant's blocks off other tenants -- nothing ever
    # empties the cache, so it grows by one block per cycle until
    # activation is refused. Measured on SDXL: 5.299 GiB after cycle 1,
    # 10.322 after cycle 2, refused on cycle 3, with the step (5.023 GiB)
    # matching the pre-warm block (5.021 GiB) to two thousandths.
    #
    # Bare activate/deactivate cycles never showed it: without a render the
    # block is reused cleanly and the pool holds flat. It takes a
    # generation in between, which is why one cycle is not a test.
    if (is.null(held)) {
        held <- tryCatch({
            s <- torch::cuda_memory_stats(device = .cuda_index(device))
            as.numeric(s$reserved_bytes$all$current) -
            as.numeric(s$allocated_bytes$all$current)
        }, error = function(e) 0)
    }
    if (!isTRUE(is.finite(held)) || held < 0) {
        held <- 0
    }
    # One target, used for both the skip and the size, so the two cannot
    # disagree. Skipping at `held >= bytes` while growing toward
    # `bytes * 1.05` put a step in the middle: 3.999 GiB held asked for
    # 0.201 GiB and 4.000 GiB held asked for nothing.
    target <- as.numeric(bytes) * 1.05
    if (held >= target) {
        return(invisible(0))
    }
    want <- target - held
    tryCatch({
        warm <- torch::torch_empty(want, dtype = torch::torch_uint8(),
                                   device = device)
        rm(warm)
        gc(verbose = FALSE)
    }, error = function(e) invisible(NULL))
    invisible(want)
}

#' Device ordinal for a torch device string
#'
#' \code{torch::cuda_memory_stats()} defaults to
#' \code{cuda_current_device()}, so reading it without an argument reports
#' whichever device happens to be current rather than the one a handle is
#' bound to. \code{resident_load()} binds an explicit \code{"cuda:N"}
#' precisely so transitions cannot drift, and a handle on \code{cuda:1}
#' deciding from \code{cuda:0}'s pool would either skip a pre-warm it needs
#' or repeat one it does not.
#'
#' @param device Character, e.g. "cuda", "cuda:0", "cuda:1".
#'
#' @return Integer ordinal. An unqualified device gives the current one.
#'
#' @keywords internal
.cuda_index <- function(device) {
    d <- as.character(device)[[1]]
    if (grepl(":", d, fixed = TRUE)) {
        n <- suppressWarnings(as.integer(sub("^.*:", "", d)))
        if (!is.na(n)) {
            return(n)
        }
    }
    tryCatch(torch::cuda_current_device(), error = function(e) 0L)
}

#' Which components a bulk activation puts on the card
#'
#' All of them, unless the family says otherwise.
#'
#' SDXL says otherwise. Its four components are only 8.0 GB pinned, so
#' bulk-onloading the set looks affordable on a 16 GB card -- and then the
#' VAE decode OOMs, because SDXL decodes 1024x1024 in float32 and that peak
#' arrives while the UNet is still resident. Measured: 8.0 GB of weights
#' plus the decode phase reached 14.38 GiB of 15.47 GiB and died asking for
#' another 512 MiB. A 12 GB card never had a chance.
#'
#' So SDXL puts only the UNet on the card and computes the text encode and
#' the decode on the host from the same pinned copies. That is the placement
#' \code{\link{auto_devices}} already recommends for this model at this tier;
#' residency's contribution is that the 5 GB UNet stops being re-read from
#' disk between models.
#'
#' @param res A resident handle.
#'
#' @return Character vector of \code{res$staging} names.
#'
#' @keywords internal
.resident_gpu_set <- function(res) {
    want <- res$gpu_components
    if (is.null(want)) {
        return(names(res$staging))
    }
    intersect(want, names(res$staging))
}

#' Refuse a bulk activation that cannot fit
#'
#' Fails before the transfer rather than part-way through it. A partial
#' onload that OOMs is recoverable (activation rolls back), but it wastes
#' the transfer and reports a libtorch allocator error instead of the
#' actual problem, which is that this model does not fit this card.
#'
#' @param res A resident handle.
#' @param free_gb Free VRAM in GB. NULL measures it. Pass a value to
#'   make the decision deterministic: with no GPU the measurement is 0,
#'   which means "cannot tell" and never refuses, so a test that wants
#'   the refusal has to state the budget rather than depend on the
#'   machine having a card.
#' @param need_bytes Bytes actually headed for the card. NULL means the
#'   whole pinned set, which is right for a family that onloads everything
#'   and wrong for one that onloads a subset -- SDXL pins 8.0 GB and sends
#'   5.1 GB of it, so charging it the full figure would refuse activations
#'   that fit.
#'
#' @return Invisibly TRUE, or an error naming both figures.
#'
#' @keywords internal
.resident_check_fits <- function(res, free_gb = NULL, need_bytes = NULL) {
    if (is.null(free_gb)) {
        free_gb <- tryCatch(.detect_vram(use_free = TRUE),
                            error = function(e) NA_real_)
    }
    if (is.null(need_bytes)) {
        need_bytes <- res$pinned_bytes
    }
    need_gb <- need_bytes / 1024 ^ 3
    if (!is.na(free_gb) && free_gb > 0 && need_gb > free_gb) {
        stop(sprintf(paste0("%s needs %.2f GB resident but only %.2f GB of ",
                            "VRAM is free. Load the pipeline with ",
                            "phase_offload = TRUE so components move on ",
                            "one phase at a time."),
                     res$model, need_gb, free_gb), call. = FALSE)
    }
    invisible(TRUE)
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
#' What activation does depends on how the pipeline was loaded:
#'
#' \itemize{
#'   \item \code{phase_offload = TRUE} (the default, and what the
#'     \code{txt2img_*} functions expect): no bulk transfer. The render
#'     moves each component on as its phase begins and back off as it
#'     ends, from these same pinned copies, so pre-loading them would be
#'     undone within one phase. Activation is the ownership claim.
#'   \item \code{phase_offload = FALSE}: every component is copied to the
#'     card up front and stays there across renders. This is the fast
#'     path, and it is checked against free VRAM first.
#' }
#'
#' The distinction is not cosmetic. FLUX.1's pinned set is 15.73 GB,
#' which does not fit a 15.47 GiB card -- bulk-onloading it OOMs even
#' though the phased render fits comfortably. So \code{state} is a claim
#' about who owns the card, not a measurement of what is on it; read
#' \code{components_on_gpu} from \code{\link{resident_status}} for the
#' measurement.
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
    # A phase-offloading pipeline moves each component onto the card as
    # its phase begins and straight back off as it ends, from these same
    # pinned copies. Bulk-onloading here is therefore redundant -- the
    # render undoes it within one phase -- and actively harmful: FLUX.1's
    # pinned set is 15.73 GB, which does not fit a 15.47 GiB card even
    # though the phased render does. For those pipelines, activation is
    # the ownership claim; the transfers stay per-phase.
    # The loader's own field wins when it kept one (the FLUX family
    # downgrades phase_offload to FALSE on a CPU device); otherwise fall
    # back to what resident_load() was asked for.
    bulk <- !isTRUE(res$pipeline$phase_offload %||% res$phase_offload)
    ok <- tryCatch({
        if (bulk) {
            onto <- .resident_gpu_set(res)
            need <- .resident_pinned_bytes(res$staging[onto])
            .resident_check_fits(res, need_bytes = need)
            .resident_prewarm(need, res$device)
            for (nm in onto) {
                .staged_onload(res$staging[[nm]], res$device)
            }
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
        # NF4 linears dequantize into a package-level scratch environment,
        # not into the module, so offloading the weights does not free it.
        # txt2vid_ltx2() deliberately SKIPS its own release when the
        # transformer is resident (the next chunk reuses the buffers), which
        # is right within a render and leaves the scratch on the card once
        # the render is over. Nothing else reclaims it: gc() and
        # cuda_empty_cache() cannot touch a block the environment still
        # references. Deactivation is the moment the handle stops owning the
        # card, so it is the moment to drop them. Unconditional on `release`
        # -- a broker that passes release = FALSE to keep the pool warm for
        # the next tenant is exactly the caller that must not be handed a
        # budget short by this scratch. The image families already release
        # it at the end of every generate.
        if (identical(res$model, "ltx")) {
            ltx23_release_dequant_buffers()
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
#'   \code{\link{txt2img_flux2}}, \code{\link{txt2img_zimage}},
#'   \code{\link{txt2vid_ltx2}} or \code{\link{txt2img_sdxl}}. For
#'   \code{sdxl} the handle supplies \code{devices} matching its own
#'   placement unless the caller names it.
#'
#' @return Whatever the family generator returns, which is always a list.
#'
#'   The five image families (\code{flux1}, \code{flux2}, \code{zimage},
#'   \code{sdxl}, \code{sd21}) return \code{list(image, metadata)}, where
#'   \code{image} is an [H, W, 3] array in [0, 1], so a caller unwraps
#'   \code{$image} uniformly across all five.
#'
#'   \code{ltx} returns \code{latents}, \code{audio_latents},
#'   \code{latent_shape} and \code{sample_rate}, plus \code{video} and
#'   \code{audio} -- but those two are produced only when
#'   \code{decode_video} and \code{decode_audio} are TRUE, which they are
#'   by default. A caller that turns either off gets a list without that
#'   field rather than a NULL one, so index it with \code{[[ ]]} and check,
#'   the way \code{\link{txt2vid_ltx2}} does internally.
#'
#'   Only the visibility differs: \code{\link{txt2img_sdxl}} and
#'   \code{\link{txt2img_sd21}} use \code{return()} while the other three
#'   image families and \code{ltx} use \code{invisible()}, which affects
#'   auto-printing at the console and nothing else.
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
                  ltx = txt2vid_ltx2,
                  sdxl = txt2img_sdxl,
                  sd21 = txt2img_sd21)
    do.call(gen, c(list(prompt, pipeline = res$pipeline),
                   .resident_gen_args(res, list(...))))
}

#' Family fixups for a resident generate call
#'
#' Split out so the SDXL device injection can be asserted without running a
#' multi-gigabyte generation.
#'
#' \code{\link{txt2img_sdxl}} does not read the pipeline's placement. With
#' its default \code{devices = "auto"} it calls \code{\link{auto_devices}}
#' afresh and moves the prompt embeds to whatever THAT returns. On a 12 GB
#' card auto can answer "unet on cuda, encoders on cpu", which contradicts a
#' bulk-activated handle whose encoders are on the card, and the text
#' encoder call then dies on a device mismatch. The handle knows where its
#' components actually are, so it says so instead of letting the generator
#' re-decide.
#'
#' Only SDXL needs this: the other families phase-offload from their own
#' pinned copies and place each component themselves as its phase begins.
#'
#' @param res A resident handle.
#' @param args The caller's \code{...}, as a list. An explicit
#'   \code{devices} wins -- this fills a gap, it does not override.
#'
#' @return \code{args}, possibly with \code{devices} added.
#'
#' @keywords internal
.resident_gen_args <- function(res, args) {
    if (res$model %in% c("sdxl", "sd21") && is.null(args$devices)) {
        on_gpu <- .resident_gpu_set(res)
        # The component set is the generator's, not the pipeline's: SD 2.1
        # declares an `encoder` (the VAE encoder img2img needs) that the
        # text-to-image pipeline never builds, and standardize_devices()
        # would otherwise refuse the list as missing a required component.
        # It follows the decoder, which is where it would live anyway.
        want <- get_required_components(res$model)
        args$devices <- stats::setNames(lapply(want, function(nm) {
            src <- if (identical(nm, "encoder")) "decoder" else nm
            if (src %in% on_gpu) {
                res$device
            } else {
                "cpu"
            }
        }), want)
    }
    ## LTX takes its text encoder per call and stores none, so a handle that
    ## loaded one has to hand it over on every generate. An explicit argument
    ## still wins -- this fills a gap rather than overriding a caller who
    ## brought precomputed embeds or a different encoder.
    if (identical(res$model, "ltx") && !is.null(res$text_encoder) &&
        is.null(args$text_encoder) && is.null(args$prompt_embeds)) {
        args$text_encoder <- res$text_encoder
        if (is.null(args$tokenizer)) {
            args$tokenizer <- res$tokenizer
        }
    }
    args
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
#'   without CUDA), \code{components_on_gpu} (how many components are
#'   *actually* resident right now), \code{loaded_at}, and
#'   \code{last_error} (NULL unless a transition failed).
#'
#'   \code{state} is the handle's claim on the card;
#'   \code{components_on_gpu} is the measurement. They disagree by
#'   design after a render on a \code{phase_offload = TRUE} pipeline,
#'   which returns each component to pinned host memory as its phase
#'   finishes: the handle stays "active" (it still owns the card's
#'   budget and can render again without touching disk) while
#'   \code{components_on_gpu} is 0. Schedule on the measurement.
#'
#' @export
resident_status <- function(res) {
    stopifnot(inherits(res, "diffuseR_resident"))
    mem <- .cuda_bytes()
    list(model = res$model, state = res$state, device = res$device,
         components = names(res$staging),
         components_on_gpu = .resident_on_gpu_count(res$staging),
         pinned_bytes = res$pinned_bytes, gpu_allocated = mem$allocated,
         gpu_reserved = mem$reserved, loaded_at = res$loaded_at,
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
        # The LTX text encoder is deliberately outside `staging` (see
        # resident_load), but it is pinned host memory this handle owns
        # and counted in pinned_bytes, so it goes the same way.
        te_staging <- attr(res$text_encoder, "staging")
        if (!is.null(te_staging)) {
            .staged_offload(te_staging)
        }
    }, error = function(e) NULL)
    res$pipeline <- NULL
    res$text_encoder <- NULL
    res$tokenizer <- NULL
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
    cat("  on gpu:     ", s$components_on_gpu, " of ", length(s$components),
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
    sprintf("%.2f GB", b / 1024 ^ 3)
}
