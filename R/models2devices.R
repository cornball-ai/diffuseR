#' Device configuration for an already-built pipeline
#'
#' The half of \code{\link{models2devices}} that does not touch the disk.
#'
#' \code{models2devices()} ends by calling \code{download_model()}, which
#' resolves the TorchScript \code{.pt} files for the model and stops with
#' "Missing model files" when they are absent. That is correct when it is
#' about to load them, and wrong when the caller already holds a pipeline:
#' a native safetensors pipeline never reads a \code{.pt}, so verifying them
#' makes a working generation depend on files it does not use. Passing
#' \code{download_models = FALSE} does not avoid it -- the check runs either
#' way and only the downloading is suppressed.
#'
#' So callers with a pipeline in hand take this path and get the same four
#' fields without the file check.
#'
#' @param model_name A character string naming the model, e.g. "sdxl".
#' @param devices A device string or named list of component devices.
#' @param unet_dtype_str A character string naming the UNet dtype, or NULL.
#'
#' @return The same shape \code{\link{models2devices}} returns:
#'   \code{devices}, \code{unet_dtype}, \code{device_cpu}, \code{device_cuda}.
#'
#' @keywords internal
.devices_for_pipeline <- function(model_name, devices, unet_dtype_str = NULL) {
    dv <- standardize_devices(devices, get_required_components(model_name))
    list(devices = dv, unet_dtype = setup_dtype(dv, unet_dtype_str),
         device_cpu = torch::torch_device("cpu"),
         device_cuda = torch::torch_device("cuda"))
}

#' models2devices
#' @description This function sets up the model directory, device configuration, and data types for diffusion models.
#' It checks the validity of the model name and devices, detects model type, and downloads the model if necessary.
#'
#' @param model_name A character string representing the name of the model to be used.
#' @param devices A character string or a named list specifying the devices for different components of the model.
#' @param unet_dtype_str A character string specifying the data type for the UNet model.
#' @param download_models Logical indicating whether to download models if they are not found.
#'
#' @return A list containing the device configuration, UNet data type, and CPU/CUDA devices.
#' @export
models2devices <- function(model_name, devices = "cpu",
                           unet_dtype_str = NULL, download_models = FALSE) {
    # Validation (same as before)
    if (is.null(model_name) || !is.character(model_name)) {
        stop("Invalid model name")
    }

    if (is.null(devices) || (!is.character(devices) && !is.list(devices))) {
        stop("Invalid devices parameter")
    }

    if (!is.null(unet_dtype_str) && !is.character(unet_dtype_str)) {
        stop("Invalid unet_dtype_str parameter")
    }

    # Get required components for this model type
    required_components <- get_required_components(model_name)

    # Standardize the 'devices' parameter
    devices <- standardize_devices(devices, required_components)

    # Set up dtype
    unet_dtype <- setup_dtype(devices, unet_dtype_str)

    device_cpu <- torch::torch_device("cpu")
    device_cuda <- torch::torch_device("cuda")

    # Verify model files are available (downloads if allowed)
    download_model(model_name, devices, unet_dtype_str,
                   download_models = download_models)

    return(list(
                devices = devices,
                unet_dtype = unet_dtype,
                device_cpu = device_cpu,
                device_cuda = device_cuda
        ))
}

#' Get required components for each model type
#' @description This function returns a list of required components for each supported model type.
#' @param model_name A character string representing the name of the model.
#' @return A character vector of required components for the specified model.
get_required_components <- function(model_name) {
    components <- list(
                       # "sd15" = c("unet", "decoder", "text_encoder", "encoder"),
                       "sd21" = c("unet", "decoder", "text_encoder", "encoder"),
                       "sdxl" = c("unet", "decoder", "text_encoder", "text_encoder2",
                                  "encoder"),
                       "flux1" = c("transformer", "decoder", "text_encoder", "text_encoder2"),
                       "flux2" = c("transformer", "decoder", "text_encoder"),
                       "zimage" = c("transformer", "decoder", "text_encoder")
                       # "sd3" = c("transformer", "decoder", "text_encoder", "text_encoder2", "text_encoder3", "encoder"),
                       # "cascade" = c("prior", "decoder", "text_encoder", "vqgan")
    )

    if (!model_name %in% names(components)) {
        stop("Unsupported model type: ", model_name)
    }

    return(components[[model_name]])
}

#' Standardize devices configuration
#' @description This function standardizes the device configuration for model components.
#' It checks if the devices parameter is a single string or a named list, and fills in missing components with reasonable defaults.
#' @param devices A character string or a named list specifying the devices for model components.
#' @param required_components A character vector of required components for the model.
#' @return A named list of devices for each required component.
standardize_devices <- function(devices, required_components) {
    if (is.character(devices) && length(devices) == 1) {
        # Single device string - apply to all components
        device_list <- as.list(rep(devices, length(required_components)))
        names(device_list) <- required_components
        return(device_list)
    }

    if (is.list(devices)) {
        # Check for required components
        missing_components <- setdiff(required_components, names(devices))

        if (length(missing_components) > 0) {
            # Try to fill in missing components with reasonable defaults
            for (component in missing_components) {
                if (component == "encoder" && "decoder" %in% names(devices)) {
                    devices$encoder <- devices$decoder
                } else if (component == "text_encoder2" &&
                    "text_encoder" %in% names(devices)) {
                    devices$text_encoder2 <- devices$text_encoder
                    message("text_encoder2 is set to text_encoder")
                } else if (component == "text_encoder3" &&
                    "text_encoder" %in% names(devices)) {
                    devices$text_encoder3 <- devices$text_encoder
                } else if (component == "transformer" &&
                    "unet" %in% names(devices)) {
                    devices$transformer <- devices$unet
                } else {
                    stop("Missing required component: ", component)
                }
            }
        }

        return(devices)
    }
    stop("'devices' must be either a single device string or a named list")
}

#' Set up dtype based on device configuration
#' @param devices A character string or a named list specifying the devices for model components.
#' @param unet_dtype_str A character string specifying the data type for the UNet model.
#' @return A torch dtype object based on the main computation device.
setup_dtype <- function(devices, unet_dtype_str) {
    # Find the main computation device (unet or transformer)
    main_device <- if ("unet" %in% names(devices)) {
        devices$unet
    } else if ("transformer" %in% names(devices)) {
        devices$transformer
    } else {
        stop("No main computation component found")
    }
    # An ordinal-qualified device picks the same dtype as the bare one.
    # resident_load() binds to an explicit "cuda:N" so later transitions
    # cannot drift, and passes that through as the component device; without
    # this, "cuda:0" matches neither branch below and falls to the "Invalid
    # device" stop.
    main_device <- sub(":.*$", "", main_device)

    if (main_device == "cpu") {
        return(torch::torch_float32())
    } else if (main_device == "cuda") {
        if (is.null(unet_dtype_str)) {
            return(torch::torch_float16())
        } else if (unet_dtype_str == "float16") {
            return(torch::torch_float16())
        } else if (unet_dtype_str == "float32") {
            return(torch::torch_float32())
        } else {
            stop("Invalid dtype: ", unet_dtype_str)
        }
    } else {
        stop("Invalid device: ", main_device)
    }
}
