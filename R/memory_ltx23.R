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
#' Selects component placement and attention chunking for the available
#' VRAM. The fp8 transformer weights (~21 GB) always stay CPU-resident
#' and stream per-linear; profiles differ in what else lives on the GPU
#' and how hard the attention is chunked.
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
    high = list(
      name = "high",
      device = "cuda",
      dtype = "bfloat16",
      pin_weights = TRUE,
      attn_chunk = 8192L,
      text_device = "cpu",
      vae_device = "cuda",
      audio_device = "cuda",
      max_resolution = c(704L, 1280L),
      max_frames = 121L
    ),
    medium = list(
      name = "medium",
      device = "cuda",
      dtype = "bfloat16",
      pin_weights = TRUE,
      attn_chunk = 4096L,
      text_device = "cpu",
      vae_device = "cuda",
      audio_device = "cuda",
      max_resolution = c(576L, 1024L),
      max_frames = 121L
    ),
    low = list(
      name = "low",
      device = "cuda",
      dtype = "bfloat16",
      pin_weights = TRUE,
      attn_chunk = 2048L,
      text_device = "cpu",
      vae_device = "cpu",
      audio_device = "cpu",
      max_resolution = c(512L, 768L),
      max_frames = 65L
    ),
    cpu_only = list(
      name = "cpu_only",
      device = "cpu",
      dtype = "float32",
      pin_weights = FALSE,
      attn_chunk = NULL,
      text_device = "cpu",
      vae_device = "cpu",
      audio_device = "cpu",
      max_resolution = c(384L, 640L),
      max_frames = 33L
    )
  )

  profiles[[profile]]
}

#' Tune the torch CUDA allocator for large-model inference
#'
#' By default torch triggers R garbage collection when CUDA allocations
#' pass ~20\% of VRAM, which fires on nearly every allocation once a
#' large model is resident and can slow inference several-fold. Sets the
#' allocator reserved rate to the actual expected footprint (clamped to
#' [0.2, 0.92]) and raises the GC call threshold. Must run BEFORE the
#' first CUDA operation of the session; user-set options are respected.
#'
#' @param footprint_gb Numeric. Expected resident GPU footprint
#'   (weights + activations).
#' @param total_gb Numeric or NULL (auto-detect total VRAM).
#'
#' @return Invisibly, the applied reserved rate (NULL if skipped).
#'
#' @export
ltx23_tune_gc <- function(footprint_gb, total_gb = NULL) {
  if (!torch::cuda_is_available()) {
    return(invisible(NULL))
  }
  if (is.null(total_gb)) {
    total_gb <- .detect_vram(use_free = FALSE)
    if (!isTRUE(total_gb > 0)) {
      return(invisible(NULL))
    }
  }

  if (is.null(getOption("torch.threshold_call_gc"))) {
    options(torch.threshold_call_gc = 16000)
  }
  rate <- NULL
  if (is.null(getOption("torch.cuda_allocator_reserved_rate"))) {
    rate <- min(0.92, max(0.20, footprint_gb / total_gb))
    options(torch.cuda_allocator_reserved_rate = rate)
  }
  invisible(rate)
}
