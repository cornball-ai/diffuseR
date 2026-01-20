#' Load a specific component of a diffusion model
#'
#' Loads a TorchScript model component (UNet, decoder, or text encoder) from the
#' local model directory, downloading it first if necessary.
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
load_model_component <- function(
  component,
  model_name = "sd21",
  device = "cpu",
  unet_dtype_str = NULL,
  download = TRUE,
  use_native = FALSE
) {

  # Set valid components based on model
  if (model_name == "sdxl") {
    valid_components <- c("unet", "decoder", "text_encoder", "text_encoder2", "encoder")
  } else {
    valid_components <- c("unet", "decoder", "text_encoder", "encoder")
  }

  # Check if component is valid
  if (!component %in% valid_components) {
    stop("Invalid component name for model '", model_name, "'. Must be one of: ",
      paste(valid_components, collapse = ", "))
  }

  # Get the model directory path
  model_dir <- file.path(tools::R_user_dir("diffuseR", "data"), model_name)

  # Check if model directory exists
  if (!dir.exists(model_dir)) {
    if (download) {
      download_result <- download_model(model_name)
      model_dir <- download_result$model_dir
    } else {
      stop("Model '", model_name,
        "' not found locally. Set download=TRUE to download.")
    }
  }

  # Determine file path for the specific component
  # Native modules can use CPU or CUDA files (moved to target device after loading)
  if (use_native) {
    # For SDXL UNet, always use CUDA file (CPU file is often corrupted/truncated)
    if (component == "unet" && model_name == "sdxl") {
      file_path <- get_component_file_path(component, model_dir, "cuda", unet_dtype_str)
    } else {
      file_path <- get_component_file_path(component, model_dir, "cpu", unet_dtype_str)
    }
  } else {
    file_path <- get_component_file_path(component, model_dir, device, unet_dtype_str)
  }

  # Check if component file exists
  if (!file.exists(file_path)) {
    if (download) {
      # Try to download the specific component
      tryCatch({
          download_component(model_name, component, device, overwrite = FALSE)
        }, error = function(e) {
          warning("Failed to download component: ", e$message)
        })

      # Check again after download attempt
      if (!file.exists(file_path)) {
        stop("Component '", component, "' could not be downloaded for model '", model_name, "'.")
      }
    } else {
      stop("Component '", component, "' not found for model '", model_name, "'. File expected at: ", file_path)
    }
  }

  # Load the model
  if (use_native && component == "unet") {
    # Native UNet (replaces TorchScript for better GPU compatibility)
    # Detect model type and use appropriate implementation
    if (model_name == "sdxl") {
      model <- unet_sdxl_native_from_torchscript(file_path, verbose = FALSE)
    } else {
      model <- unet_native_from_torchscript(file_path, verbose = FALSE)
    }
    # Handle dtype: use float16 for GPU unless explicitly set to float32
    if (device == "cuda" && (is.null(unet_dtype_str) || unet_dtype_str == "float16")) {
      model$to(device = torch::torch_device(device), dtype = torch::torch_float16())
    } else {
      model$to(device = torch::torch_device(device))
    }
  } else if (use_native && component == "decoder") {
    # Native decoder (replaces TorchScript for better GPU compatibility)
    model <- vae_decoder_native()
    load_decoder_weights(model, file_path, verbose = FALSE)
    model$to(device = torch::torch_device(device))
  } else if (use_native && component == "text_encoder") {
    # Native text encoder - detect architecture from weights
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
    # Native text encoder 2 - detect architecture from weights
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
    # TorchScript (legacy)
    model <- torch::jit_load(file_path, map_location = torch::torch_device(device))
  }

  return(model)
}

# Helper function to determine file path
get_component_file_path <- function(
  component,
  model_dir,
  device,
  unet_dtype_str
) {
  if (component == "unet") {
    if (device == "cpu") {
      file_path <- file.path(model_dir, paste0(component, "-", device, ".pt"))
    } else {
      # Handle GPU unet with dtype
      if (is.null(unet_dtype_str) || unet_dtype_str == "float16") {
        dtype <- "float16"
      } else if (unet_dtype_str == "float32") {
        dtype <- "float32"
      } else {
        stop("Invalid unet_dtype_str: must be 'float16' or 'float32'")
      }
      file_path <- file.path(model_dir, paste0("unet-cuda-", dtype, ".pt"))
    }
  } else {
    # All other components (decoder, text_encoder, text_encoder1, text_encoder2, encoder)
    file_path <- file.path(model_dir, paste0(component, "-", device, ".pt"))
  }

  return(file_path)
}

# Convenience function to load both text encoders for SDXL
load_text_encoders <- function(
  model_name = "sdxl",
  device = "cpu",
  download = TRUE
) {
  if (model_name != "sdxl") {
    # For non-SDXL models, return single text encoder
    return(list(
        text_encoder = load_model_component("text_encoder", model_name, device, download = download)
      ))
  }

  # For SDXL, load both text encoders
  text_encoder <- load_model_component("text_encoder", model_name, device, download = download)
  text_encoder2 <- load_model_component("text_encoder2", model_name, device, download = download)

  return(list(
      text_encoder = text_encoder,
      text_encoder2 = text_encoder2
    ))
}

