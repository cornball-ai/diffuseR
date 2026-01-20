#' Auto-Configure Device Assignment
#'
#' Automatically determines optimal device configuration for diffusion model
#' components based on available VRAM and GPU architecture. Uses gpuctl for
#' detection if available, otherwise falls back to sensible defaults.
#'
#' @param model Character. Model type: "sd21" or "sdxl".
#' @param strategy Character. Memory strategy: "auto" (default), "full_gpu",
#'   "unet_gpu", or "cpu_only". See Details.
#'
#' @return A named list of device assignments suitable for `models2devices()`.
#'
#' @details
#' Strategies:
#' \describe{
#'   \item{"auto"}{Detect VRAM and choose best strategy (requires gpuctl)}
#'   \item{"full_gpu"}{All components on CUDA (16GB+ VRAM for SDXL)}
#'   \item{"unet_gpu"}{Only unet on CUDA, rest on CPU (8GB+ VRAM)}
#'   \item{"cpu_only"}{All components on CPU}
#' }
#'
#' If gpuctl is not installed, "auto" falls back to "unet_gpu" which works on
#' most modern GPUs (8GB+ VRAM).
#'
#' On Blackwell GPUs (RTX 50xx), "unet_gpu" is forced due to TorchScript
#' compatibility issues, regardless of available VRAM.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Auto-detect best configuration
#' devices <- auto_devices("sdxl")
#'
#' # Use with models2devices
#' m2d <- models2devices("sdxl", devices = auto_devices("sdxl"))
#'
#' # Force CPU-only
#' devices <- auto_devices("sdxl", strategy = "cpu_only")
#' }
auto_devices <- function(
  model = "sdxl",
  strategy = "auto"
) {
  # If gpuctl is available, use it

  if (requireNamespace("gpu.ctl", quietly = TRUE)) {
    return(gpu.ctl::recommended_devices(model = model, strategy = strategy))
  }

  # Fallback when gpuctl not available
  if (strategy == "auto") {
    message("gpuctl not installed - using unet_gpu strategy as default")
    message("Install gpuctl for auto-detection: install.packages('gpuctl')")
    strategy <- "unet_gpu"
  }

  # Build device config manually
  .build_fallback_devices(model, strategy)
}

#' Build fallback device configuration
#'
#' @param model Character. Model type.
#' @param strategy Character. Memory strategy.
#' @return Named list of device assignments.
#' @keywords internal
.build_fallback_devices <- function(
  model,
  strategy
) {
  # Components by model
  components <- list(
    sd21 = c("unet", "decoder", "text_encoder", "encoder"),
    sdxl = c("unet", "decoder", "text_encoder", "text_encoder2", "encoder")
  )

  if (!model %in% names(components)) {
    stop("Unsupported model: ", model)
  }

  comp <- components[[model]]

  if (strategy == "full_gpu") {
    devices <- as.list(rep("cuda", length(comp)))
    names(devices) <- comp
  } else if (strategy == "unet_gpu") {
    devices <- as.list(rep("cpu", length(comp)))
    names(devices) <- comp
    devices$unet <- "cuda"
  } else {
    # cpu_only
    devices <- as.list(rep("cpu", length(comp)))
    names(devices) <- comp
  }

  devices
}

