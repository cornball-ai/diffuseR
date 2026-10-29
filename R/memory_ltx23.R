#' LTX-2.3 Memory Profiles and CUDA GC Tuning
#'
#' Memory management for running the 22B LTX-2.3 transformer on limited
#' VRAM, built on the patterns proven in the whisper and chatterbox
#' packages: torch allocator GC tuning before the first CUDA op, fp8
#' CPU-resident streaming weights, query-chunked attention, and
#' phase-sequential component placement.
#'
#' @name memory_ltx23
NULL

#' Get an LTX-2.3 memory profile
#'
#' Selects transformer precision, component placement, and attention
#' chunking for the available VRAM. Measured on an RTX 5060 Ti (16 GB):
#' fp8 streaming peaks ~11.6 GB (without phase offloading) at
#' 512x320x49; NF4 keeps the whole 22B transformer resident (~12.5 GB)
#' and removes the ~21 GB/step PCIe weight streaming. The NF4 profile
#' renders 1280x704x121 with audio in ~23 min at a 15.7 GB peak
#' (tiled VAE decode, in-place feed-forward GELU, and the default
#' \code{diffuseR.attn_budget} of 1.5e8 all required at that size).
#'
#' Below 14 GB the NF4 transformer keeps \code{resident_gb} on the GPU
#' and streams its remaining blocks from pinned host memory every step
#' (pass it to \code{\link{ltx23_load_pipeline}}), and below 12 GB the
#' Gemma 4 encoder streams its layers (\code{stream_text}; pass it as
#' \code{stream} to \code{\link{load_gemma4_text_encoder}}). Measured
#' with the rest of a 16 GB card held back: LTX-2.5 768x512x121 renders
#' in ~84 s on both 12 GB (8 GB resident) and 8 GB (4 GB resident), the
#' same as on the full card.
#'
#' \describe{
#'   \item{precision "nf4"}{Weights resident on the GPU (or partly
#'     streamed); fastest steps; about 9 percent weight round-trip error.}
#'   \item{precision "fp8"}{Weights CPU-resident, streamed per linear;
#'     near-bf16 quality; each step pays the PCIe transfer.}
#' }
#'
#' Host RAM decides two more fields for the GPU tiers. \code{split_text}
#' asks the caller to encode in its own phase (Gemma, then the
#' connectors, each freed before the next loads; see
#' \code{\link{ltx23_text_embeds}}) instead of holding the encoder next
#' to the render pipeline. \code{resident_from} is passed to
#' \code{\link{ltx23_load_pipeline}}: \code{"disk"} drops the host copy
#' of the GPU-resident transformer blocks. Below 32 GB available the
#' text phase splits off; below 22 GB the resident blocks come from
#' disk (measured on LTX-2.5 at 768x512x121 with an 8 GB card: the
#' render phase peaks at 20.1 GB of host RAM with host copies and 16.0
#' GB without, for ~8 s more per render). Below 18 GB,
#' \code{stage_from = "disk"} (for \code{\link{ltx23_load_pipeline}} and
#' \code{\link{load_gemma4_nf4}}) keeps no weights in host memory: every
#' phase reads them from the memory-mapped artifacts. Measured the same
#' way under a 10 GB memory cap that also counts page cache: the text
#' phase holds 1.5 GB of anonymous memory, the render phase at most
#' 4.6 GB, and a warm render takes 127 s instead of 82 s.
#'
#' @param vram_gb Numeric or NULL (auto-detect free VRAM).
#' @param host_ram_gb Numeric or NULL (auto-detect available RAM; NA
#'   when undetectable, which keeps the all-in-RAM defaults).
#'
#' @return Named list with device/dtype placement, \code{attn_chunk},
#'   \code{pin_weights}, \code{resident_gb}, \code{stream_text},
#'   \code{split_text}, \code{resident_from}, \code{stage_from}, and
#'   resolution caps.
#'
#' @export
ltx23_memory_profile <- function(vram_gb = NULL, host_ram_gb = NULL) {
    if (is.null(vram_gb)) {
        vram_gb <- .detect_vram(use_free = TRUE)
        message(sprintf("Detected %.1f GB free VRAM", vram_gb))
    }
    if (is.null(host_ram_gb)) {
        host_ram_gb <- .detect_host_ram()
    }
    ram_known <- isTRUE(is.finite(host_ram_gb))

    profile <- if (vram_gb >= 14) {
        "high"
    } else if (vram_gb >= 10) {
        "medium"
    } else if (vram_gb >= 7) {
        "low"
    } else {
        "cpu_only"
    }

    # Streamed tiers keep vram_gb - 4 GB of the transformer resident:
    # 768x512x121 activations, connectors and decode fit in the rest
    resident <- max(0, vram_gb - 4)
    profiles <- list(
                     high = list(name = "high", device = "cuda", dtype = "bfloat16",
                                 precision = "nf4", phase_offload = TRUE,
                                 pin_weights = FALSE, attn_chunk = NULL,
                                 text_device = "cpu", resident_gb = Inf,
                                 stream_text = FALSE,
                                 max_resolution = c(704L, 1280L),
                                 max_frames = 121L),
                     medium = list(
                                   name = "medium",
                                   device = "cuda",
                                   dtype = "bfloat16",
                                   precision = "nf4",
                                   phase_offload = TRUE,
                                   pin_weights = TRUE,
                                   attn_chunk = NULL,
                                   text_device = "cuda",
                                   resident_gb = resident,
                                   stream_text = vram_gb < 12,
                                   max_resolution = c(512L, 768L),
                                   max_frames = 121L
        ),
                     low = list(
                                name = "low",
                                device = "cuda",
                                dtype = "bfloat16",
                                precision = "nf4",
                                phase_offload = TRUE,
                                pin_weights = TRUE,
                                attn_chunk = NULL,
                                text_device = "cuda",
                                resident_gb = resident,
                                stream_text = TRUE,
                                max_resolution = c(512L, 768L),
                                max_frames = 121L
        ),
                     cpu_only = list(
                                     name = "cpu_only",
                                     device = "cpu",
                                     dtype = "float32",
                                     precision = "fp8",
                                     phase_offload = FALSE,
                                     pin_weights = FALSE,
                                     attn_chunk = NULL,
                                     text_device = "cpu",
                                     resident_gb = Inf,
                                     stream_text = FALSE,
                                     max_resolution = c(384L, 640L),
                                     max_frames = 33L
        )
    )

    out <- profiles[[profile]]
    gpu <- out$device == "cuda"
    out$split_text <- gpu && ram_known && host_ram_gb < .ltx23_ram_one_process
    out$resident_from <- if (gpu && ram_known &&
                             host_ram_gb < .ltx23_ram_host_resident) {
        "disk"
    } else {
        "host"
    }
    out$stage_from <- if (gpu && ram_known &&
                          host_ram_gb < .ltx23_ram_host_staged) {
        "disk"
    } else {
        "host"
    }
    out
}

# Host RAM (GB) for the LTX GPU tiers: the render pipeline plus the
# Gemma 4 encoder in one process, the render pipeline alone with its
# resident blocks kept in host memory, and with only the streamed
# blocks kept there (LTX-2.5 768x512x121 on a simulated 8 GB card,
# high-water marks: 20.1 GB with host copies, 16.0 GB re-reading the
# resident part from disk, 12.4 GB for the split text phase). Below
# the last, stage_from = "disk" keeps no weights in host memory.
.ltx23_ram_one_process <- 32
.ltx23_ram_host_resident <- 22
.ltx23_ram_host_staged <- 18

#' Tune the torch CUDA allocator for large-resident inference
#'
#' Stops the allocator GC storm (see the "Performance Levers" vignette
#' and chatterbox's "performance" vignette): lantern proactively calls R's gc()
#' whenever reserved memory exceeds \code{torch.cuda_allocator_reserved_rate}
#' (default 0.20) of the card. With ~75\% of VRAM occupied by resident
#' weights that fires on nearly every allocation. Raising the rate to the
#' actual footprint is safe here because the LTX hot loops compute into
#' persistent scratch buffers (near-zero per-step garbage). Also raises
#' the host-allocation GC threshold and defaults
#' \code{PYTORCH_CUDA_ALLOC_CONF} to expandable segments. User-set
#' options win.
#'
#' \code{start_torch()} reads the gate options exactly once, so setting
#' them after torch has started is inert on its own. The three CUDA
#' gates are therefore also pushed into the live allocator here (the
#' \code{.flux_gc_gates} pattern), which makes this function effective
#' whenever it runs. The host-side \code{torch.threshold_call_gc} has
#' no live setter; the package defaults it in \code{.onLoad} so torch
#' reads it at init in any session that loads diffuseR before running
#' torch ops.
#'
#' @param footprint_gb Numeric. Expected resident GPU footprint in GB
#'   (NF4 transformer: ~12).
#' @param total_gb Numeric or NULL (auto-detect total VRAM).
#'
#' @return Invisibly, the applied reserved rate (NULL if skipped).
#'
#' @export
ltx23_tune_gc <- function(footprint_gb = 12, total_gb = NULL) {
    if (!nzchar(Sys.getenv("PYTORCH_CUDA_ALLOC_CONF"))) {
        Sys.setenv(PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True")
    }
    # Options are set before the first torch:: call in this function:
    # torch::cuda_is_available() below starts torch, and start_torch
    # reads these once. (The old order set them one call after init
    # read them - inert by construction.)
    if (is.null(getOption("torch.threshold_call_gc"))) {
        options(torch.threshold_call_gc = 16000)
    }
    # The other two allocator-callback gates (defaults 0.8): measured
    # 32.7s -> 21.5s on the tiled decode under expandable segments, and
    # R-gc share 50% -> 32% on the native backend, with no wall-time
    # downside in any condition
    if (is.null(getOption("torch.cuda_allocator_allocated_rate"))) {
        options(torch.cuda_allocator_allocated_rate = 0.95)
    }
    if (is.null(getOption("torch.cuda_allocator_allocated_reserved_rate"))) {
        options(torch.cuda_allocator_allocated_reserved_rate = 0.95)
    }
    rate <- NULL
    if (is.null(getOption("torch.cuda_allocator_reserved_rate"))) {
        if (is.null(total_gb)) {
            total_gb <- .detect_vram(use_free = FALSE)
        }
        if (isTRUE(total_gb > 0)) {
            rate <- min(0.92, max(0.20, footprint_gb / total_gb))
            options(torch.cuda_allocator_reserved_rate = rate)
        }
    }
    if (!torch::cuda_is_available()) {
        return(invisible(rate))
    }
    # Torch is usually long started by the time a loader calls this:
    # push the CUDA gates into the live allocator directly.
    push <- get0("cpp_set_cuda_allocator_allocator_thresholds",
                 envir = asNamespace("torch"))
    if (is.function(push)) {
        try(push(
                 getOption("torch.cuda_allocator_reserved_rate", 0.2),
                 getOption("torch.cuda_allocator_allocated_rate", 0.8),
                 getOption("torch.cuda_allocator_allocated_reserved_rate", 0.8)
            ), silent = TRUE)
    }
    invisible(rate)
}
