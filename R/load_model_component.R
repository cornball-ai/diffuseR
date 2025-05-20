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
#'
#' @return A torch model object.
#' @export
#'
#' @examples
#' \dontrun{
#' unet <- load_model_component("unet", "stable-diffusion-2-1", "cpu")
#' }
load_model_component <- function(component, 
                                 model_name = "stable-diffusion-2-1", 
                                 device = "cpu",
                                 unet_dtype_str = NULL,
                                 download = TRUE) {
  
  # Set of valid components
  valid_components <- c("unet", "decoder", "text_encoder", "encoder")
  
  # Check if component is valid
  if (!component %in% valid_components) {
    stop("Invalid component name. Must be one of: ", 
         paste(valid_components, collapse = ", "))
  }
  
  # Get the model directory path
  model_dir <- file.path(tools::R_user_dir("diffuseR", "data"), model_name)
  
  # Check if model directory exists
  if (!dir.exists(model_dir)) {
    if (download) {
      model_dir <- download_model(model_name)
    } else {
      stop("Model '", model_name,
           "' not found locally. Set download=TRUE to download.")
    }
  }
  
  # File path for the specific component
  if(component != "unet"){
    file_path <- file.path(model_dir, paste0(component, "-", device, ".pt"))
  } else {
    if(component == "unet" & device == "cpu"){
      file_path <- file.path(model_dir, paste0(component, "-", device, ".pt"))
    } else {
      if(is.null(unet_dtype_str) | unet_dtype_str == "float16"){
        file_path <- file.path(model_dir, "unet-cuda-float16.pt")
      } else {
        if(unet_dtype_str == "float32"){
          file_path <- file.path(model_dir, "unet-cuda-float32.pt")
        } else {
          stop("Invalid unet_dtype_str or component")
        }
      }
    }
  }
  
  # Check if component file exists
  if (!file.exists(file_path)) {
    if (download) {
      download_component(model_name, component, device, overwrite = FALSE)
      # Check again after download attempt
      if (!file.exists(file_path)) {
        stop("Component '", component, "' could not be downloaded.")
      }
    } else {
      stop("Component '", component, "' not found for model '", model_name, "'.")
    }
  }
  
  # Load the model with torch
  model <- torch::jit_load(file_path, map_location = torch::torch_device(device))
  
  return(model)
}
