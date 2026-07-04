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
#' and removes the ~21 GB/step PCIe weight streaming.
#'
#' \describe{
#'   \item{precision "nf4"}{Weights resident on the GPU; fastest steps;
#'     ~9\% weight round-trip error.}
#'   \item{precision "fp8"}{Weights CPU-resident, streamed per linear;
#'     near-bf16 quality; each step pays the PCIe transfer.}
#' }
#'
#' @param vram_gb Numeric or NULL (auto-detect free VRAM).
#'
#' @return Named list with device/dtype placement, \code{attn_chunk},
#'   \code{pin_weights}, and resolution caps.
#'
#' @export
ltx23_memory_profile <- function(vram_gb = NULL) {
    if (is.null(vram_gb)) {
        vram_gb <- .detect_vram(use_free = TRUE)
        message(sprintf("Detected %.1f GB free VRAM", vram_gb))
    }

    profile <- if (vram_gb >= 14) {
        "high"
    } else if (vram_gb >= 10) {
        "medium"
    } else if (vram_gb >= 7) {
        "low"
    } else {
        "cpu_only"
    }

    profiles <- list(
                     high = list(name = "high", device = "cuda", dtype = "bfloat16",
                                 precision = "nf4", phase_offload = TRUE,
                                 pin_weights = FALSE, attn_chunk = NULL,
                                 text_device = "cpu", max_resolution = c(512L, 768L),
                                 max_frames = 121L),
                     medium = list(
                                   name = "medium",
                                   device = "cuda",
                                   dtype = "bfloat16",
                                   precision = "fp8",
                                   phase_offload = TRUE,
                                   pin_weights = TRUE,
                                   attn_chunk = 4096L,
                                   text_device = "cpu",
                                   max_resolution = c(576L, 1024L),
                                   max_frames = 121L
        ),
                     low = list(
                                name = "low",
                                device = "cuda",
                                dtype = "bfloat16",
                                precision = "fp8",
                                phase_offload = TRUE,
                                pin_weights = TRUE,
                                attn_chunk = 2048L,
                                text_device = "cpu",
                                max_resolution = c(512L, 768L),
                                max_frames = 65L
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
                                     max_resolution = c(384L, 640L),
                                     max_frames = 33L
        )
    )

    profiles[[profile]]
}

#' Tune the torch CUDA allocator for large-resident inference
#'
#' Configures the torch CUDA allocator for pipelines that keep most of
#' the GPU occupied by resident weights (cf. the mlverse torch
#' memory-management article). \code{torch.cuda_allocator_reserved_rate}
#' is deliberately left at its default: raising it (a common recipe for
#' small-resident models) disables R garbage collection while reserved
#' memory is below the rate, which starves garbage-heavy inference. This
#' instead lowers \code{torch.cuda_allocator_allocated_rate} so full
#' collections engage earlier under pressure, and defaults
#' \code{PYTORCH_CUDA_ALLOC_CONF} to expandable segments (must run
#' before the first CUDA allocation). User-set options are respected.
#'
#' @param allocated_rate Numeric. Allocated/total ratio above which the
#'   allocator requests a full R collection (torch default 0.8).
#'
#' @return Invisibly, NULL.
#'
#' @export
ltx23_tune_gc <- function(allocated_rate = 0.7) {
    if (!nzchar(Sys.getenv("PYTORCH_CUDA_ALLOC_CONF"))) {
        Sys.setenv(PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True")
    }
    if (is.null(getOption("torch.cuda_allocator_allocated_rate"))) {
        options(torch.cuda_allocator_allocated_rate = allocated_rate)
    }
    invisible(NULL)
}
