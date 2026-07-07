#' FLUX Memory Profiles
#'
#' VRAM-based execution profiles for the FLUX.1-schnell pipeline,
#' following the LTX-2.3 profile pattern. The 12B transformer runs NF4
#' (~7 GB, GPU-resident) or fp8 (~12 GB, CPU-resident and streamed);
#' the T5-XXL text encoder runs float32 on the CPU by default.
#'
#' @name memory_flux
NULL

#' Resolve a FLUX memory profile
#'
#' @param vram_gb Numeric or NULL. Available VRAM; auto-detected when
#'   NULL (via gpu.ctl or nvidia-smi).
#'
#' @return List with \code{name}, \code{precision} ("nf4"/"fp8"),
#'   \code{attn_chunk}, \code{text_device}, \code{phase_offload}, and
#'   \code{max_pixels} (largest validated image area).
#'
#' @export
flux_memory_profile <- function(vram_gb = NULL) {
    if (is.null(vram_gb)) {
        vram_gb <- .detect_vram(use_free = TRUE)
        if (is.null(vram_gb) || is.na(vram_gb) || vram_gb <= 0) {
            vram_gb <- 0
        }
    }

    if (vram_gb >= 12) {
        list(name = "high", precision = "nf4", attn_chunk = NULL,
             text_device = "cpu", phase_offload = TRUE,
             max_pixels = 1536L * 1536L)
    } else if (vram_gb >= 9) {
        list(name = "medium", precision = "nf4", attn_chunk = 2048L,
             text_device = "cpu", phase_offload = TRUE,
             max_pixels = 1024L * 1024L)
    } else if (vram_gb >= 7) {
        list(name = "low", precision = "fp8", attn_chunk = 1024L,
             text_device = "cpu", phase_offload = TRUE,
             max_pixels = 768L * 768L)
    } else {
        list(name = "cpu_only", precision = "nf4", attn_chunk = NULL,
             text_device = "cpu", phase_offload = FALSE,
             max_pixels = 512L * 512L)
    }
}
