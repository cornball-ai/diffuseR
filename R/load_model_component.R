#' Load a specific component of a diffusion model
#'
#' Loads a TorchScript model component (UNet, decoder, or text encoder) from the
#' hfhub cache or legacy model directory, downloading it first if necessary.
#'
#' @param component Character string, the component to load: "unet", "decoder", or "text_encoder".
#' @param model_name Character string, the name of the model to use.
#' @param device Character string, the torch device to load the model onto ("cpu" or "cuda").
#' @param unet_dtype_str Optional; the data type for the UNet model. If `NULL`, defaults to `float32` for CPU and `float16` for CUDA.
#' @param download Logical; if `TRUE` (default), downloads the model if it doesn't exist locally.
#' @param use_native Logical; if `TRUE`, uses native R torch modules instead of TorchScript.
#'   Supported for unet, decoder, text_encoder, and text_encoder2. Native modules have
#'   better GPU compatibility (especially on Blackwell/RTX 50xx).
#'
#' @return A torch model object.
#' @export
#'
#' @examples
#' \dontrun{
#' unet <- load_model_component("unet", "sd21", "cpu")
#' }
load_model_component <- function (component, model_name = "sd21",
                                  device = "cpu", unet_dtype_str = NULL,
                                  download = TRUE, use_native = FALSE) {
    # Set valid components based on model
    if (model_name == "sdxl") {
        valid_components <- c("unet", "decoder", "text_encoder", "text_encoder2", "encoder")
    } else {
        valid_components <- c("unet", "decoder", "text_encoder", "encoder")
    }

    if (!component %in% valid_components) {
        stop("Invalid component name for model '", model_name, "'. Must be one of: ",
            paste(valid_components, collapse = ", "))
    }

    # Determine filename for this component
    if (use_native) {
        # For SDXL UNet, always use CUDA file (CPU file is often corrupted/truncated)
        if (component == "unet" && model_name == "sdxl") {
            filename <- component_filename("unet", "cuda", unet_dtype_str)
        } else {
            filename <- component_filename(component, "cpu", unet_dtype_str)
        }
    } else {
        filename <- component_filename(component, device, unet_dtype_str)
    }

    # Resolve file path via hfhub (cache, legacy, or download)
    file_path <- hf_download_pt(model_name, filename, download = download)

    # Load the model
    if (use_native && component == "unet") {
        if (model_name == "sdxl") {
            model <- unet_sdxl_native_from_torchscript(file_path, verbose = FALSE)
        } else {
            model <- unet_native_from_torchscript(file_path, verbose = FALSE)
        }
        if (device == "cuda" && (is.null(unet_dtype_str) || unet_dtype_str == "float16")) {
            model$to(device = torch::torch_device(device), dtype = torch::torch_float16())
        } else {
            model$to(device = torch::torch_device(device))
        }
    } else if (use_native && component == "decoder") {
        model <- vae_decoder_native()
        load_decoder_weights(model, file_path, verbose = FALSE)
        model$to(device = torch::torch_device(device))
    } else if (use_native && component == "text_encoder") {
        arch <- detect_text_encoder_architecture(file_path)
        model <- text_encoder_native(
            vocab_size = arch$vocab_size,
            context_length = arch$context_length,
            embed_dim = arch$embed_dim,
            num_layers = arch$num_layers,
            num_heads = arch$num_heads,
            mlp_dim = arch$mlp_dim,
            apply_final_ln = arch$apply_final_ln
        )
        load_text_encoder_weights(model, file_path, verbose = FALSE)
        model$to(device = torch::torch_device(device))
    } else if (use_native && component == "text_encoder2") {
        arch <- detect_text_encoder_architecture(file_path)
        model <- text_encoder2_native(
            vocab_size = arch$vocab_size,
            context_length = arch$context_length,
            embed_dim = arch$embed_dim,
            num_layers = arch$num_layers,
            num_heads = arch$num_heads,
            mlp_dim = arch$mlp_dim
        )
        load_text_encoder2_weights(model, file_path, verbose = FALSE)
        model$to(device = torch::torch_device(device))
    } else {
        model <- torch::jit_load(file_path, map_location = torch::torch_device(device))
    }

    model
}

# Build the expected filename for a component
component_filename <- function (component, device, unet_dtype_str = NULL) {
    if (component == "unet" && device != "cpu") {
        dtype <- if (is.null(unet_dtype_str) || unet_dtype_str == "float16") {
            "float16"
        } else if (unet_dtype_str == "float32") {
            "float32"
        } else {
            stop("Invalid unet_dtype_str: must be 'float16' or 'float32'")
        }
        paste0("unet-cuda-", dtype, ".pt")
    } else {
        paste0(component, "-", device, ".pt")
    }
}

# Convenience function to load both text encoders for SDXL
load_text_encoders <- function (model_name = "sdxl", device = "cpu",
                                download = TRUE) {
    if (model_name != "sdxl") {
        return(list(
                text_encoder = load_model_component("text_encoder", model_name, device, download = download)
            ))
    }

    text_encoder <- load_model_component("text_encoder", model_name, device, download = download)
    text_encoder2 <- load_model_component("text_encoder2", model_name, device, download = download)

    list(
        text_encoder = text_encoder,
        text_encoder2 = text_encoder2
    )
}

