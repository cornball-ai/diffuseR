#' Get SDXL Memory Profile
#'
#' Determines optimal memory configuration for SDXL image generation
#' based on available VRAM.
#'
#' @param vram_gb Numeric. Available VRAM in GB, or NULL for auto-detection.
#'
#' @return A list with memory profile settings.
#'
#' @details
#' Memory profiles for SDXL:
#' \describe{
#'   \item{full_gpu}{16GB+ - All components on CUDA}
#'   \item{balanced}{10-12GB - UNet + decoder on CUDA, text encoders on CPU}
#'   \item{unet_gpu}{6-10GB - Only UNet on CUDA, everything else CPU}
#'   \item{cpu_only}{<6GB - All on CPU}
#' }
#'
#' Each profile also specifies:
#' - cfg_mode: "batched" or "sequential" (sequential halves peak memory)
#' - cleanup: "none", "phase", or "step" (when to clear VRAM)
#' - dtype: "float16" or "float32"
#' - max_resolution: maximum image dimension
#'
#' @export
#'
#' @examples
#' # A stated VRAM budget is deterministic and needs no GPU.
#' str(sdxl_memory_profile(vram_gb = 8))
#'
#' str(sdxl_memory_profile(vram_gb = 24))
#'
#' # Auto-detect free VRAM on this machine.
#' str(sdxl_memory_profile())
sdxl_memory_profile <- function(vram_gb = NULL) {
    # Auto-detect free VRAM if not provided
    if (is.null(vram_gb)) {
        vram_gb <- .detect_vram(use_free = TRUE)
        message(sprintf("Detected %.1f GB free VRAM", vram_gb))
    }

    # Determine profile level
    if (vram_gb >= 16) {
        profile <- "full_gpu"
    } else if (vram_gb >= 10) {
        profile <- "balanced"
    } else if (vram_gb >= 6) {
        profile <- "unet_gpu"
    } else {
        profile <- "cpu_only"
    }

    # Build profile config
    profiles <- list(
                     full_gpu = list(
                                     name = "full_gpu",
                                     devices = list(unet = "cuda", decoder = "cuda",
                text_encoder = "cuda", text_encoder2 = "cuda",
                encoder = "cuda"),
                                     dtype = "float16",
                                     cfg_mode = "batched",
                                     cleanup = "none",
                                     max_resolution = 1536L,
                                     step_cleanup_interval = 0L # No step cleanup
        ),
                     balanced = list(
                                     name = "balanced",
                                     devices = list(
                unet = "cuda",
                decoder = "cuda",
                text_encoder = "cpu",
                text_encoder2 = "cpu",
                encoder = "cpu"
            ),
                                     dtype = "float16",
                                     cfg_mode = "batched",
                                     cleanup = "phase", # Cleanup between text encoding and denoising
                                     max_resolution = 1024L,
                                     step_cleanup_interval = 0L
        ),
                     unet_gpu = list(
                                     name = "unet_gpu",
                                     devices = list(
                unet = "cuda",
                decoder = "cpu",
                text_encoder = "cpu",
                text_encoder2 = "cpu",
                encoder = "cpu"
            ),
                                     dtype = "float16",
                                     cfg_mode = "sequential", # Sequential CFG halves peak memory
                                     cleanup = "phase",
                                     max_resolution = 1024L,
                                     step_cleanup_interval = 10L # Cleanup every 10 steps
        ),
                     cpu_only = list(
                                     name = "cpu_only",
                                     devices = list(
                unet = "cpu",
                decoder = "cpu",
                text_encoder = "cpu",
                text_encoder2 = "cpu",
                encoder = "cpu"
            ),
                                     dtype = "float32", # CPU often faster with float32
                                     cfg_mode = "sequential",
                                     cleanup = "none", # No GPU to clean
                                     max_resolution = 768L,
                                     step_cleanup_interval = 0L
        )
    )

    profiles[[profile]]
}
