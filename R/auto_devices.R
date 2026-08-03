#' Auto-Configure Device Assignment
#'
#' Automatically determines optimal device configuration for diffusion model
#' components based on available VRAM (via nvidia-smi) and GPU architecture.
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
#'   \item{"auto"}{Detect free VRAM and choose the best strategy}
#'   \item{"full_gpu"}{All components on CUDA (10GB+ free VRAM for SDXL)}
#'   \item{"unet_gpu"}{Only unet on CUDA, rest on CPU (6GB+ for SDXL)}
#'   \item{"cpu_only"}{All components on CPU}
#' }
#'
#' On Blackwell GPUs (RTX 50xx), "unet_gpu" is forced due to TorchScript
#' compatibility issues, regardless of available VRAM. The native modules
#' (`use_native_unet` and friends) do not have this restriction.
#'
#' @export
#'
#' @examples
#' # Force a strategy: no GPU or nvidia-smi needed.
#' str(auto_devices("sdxl", strategy = "cpu_only"))
#'
#' str(auto_devices("sd21", strategy = "unet_gpu"))
#'
#' # Auto-detect free VRAM and pick a strategy for this machine.
#' str(auto_devices("sdxl"))
auto_devices <- function(model = "sdxl", strategy = "auto") {
    # Free-VRAM requirements in GB (float16 component sizes + overhead)
    requirements <- list(sd21 = list(full_gpu = 4, unet_gpu = 3),
                         sdxl = list(full_gpu = 10, unet_gpu = 6))
    req <- requirements[[model]]
    if (is.null(req)) {
        stop("Unsupported model: ", model, ". Supported: ",
             paste(names(requirements), collapse = ", "))
    }

    if (strategy == "auto") {
        vram <- .detect_vram(use_free = TRUE)
        strategy <- if (is_blackwell_gpu()) {
            "unet_gpu"
        } else if (vram >= req$full_gpu) {
            "full_gpu"
        } else if (vram >= req$unet_gpu) {
            "unet_gpu"
        } else {
            "cpu_only"
        }
        message(sprintf("auto_devices: %s (%.1f GB free VRAM)", strategy, vram))
    } else if (identical(strategy, "full_gpu") && is_blackwell_gpu()) {
        # TorchScript workaround: full_gpu is not supported on Blackwell
        message("Blackwell GPU detected - overriding full_gpu to unet_gpu")
        strategy <- "unet_gpu"
    }

    .build_fallback_devices(model, strategy)
}

#' Build fallback device configuration
#'
#' @param model Character. Model type.
#' @param strategy Character. Memory strategy.
#' @return Named list of device assignments.
#' @keywords internal
.build_fallback_devices <- function(model, strategy) {
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
