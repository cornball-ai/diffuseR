#` load_pipeline` function to load a diffusion model pipeline
#' Load a diffusion model pipeline
#' 
#' This function loads a diffusion model pipeline consisting of a UNet, VAE decoder, and text encoder.
#' It initializes the models and sets up the environment for inference.
#' 
#' @param model_name The name of the model to load.
#' @param m2d A list containing model-to-device mappings and configurations.
#' @param i2i Logical indicating whether to load the encoder for img2img().
#' @param unet_dtype_str A string representing the data type for the UNet model (e.g., "float32", "float16").
#' @param ... Additional arguments passed to the model loading functions.
#' 
#' @return An environment containing the loaded models and configuration.
#' @export
#' 
#' @examples
#' \dontrun{
#' pipeline <- load_pipeline("my_model", device = "cuda")
#' }
#' 
load_pipeline <- function(model_name, m2d, i2i = FALSE, unet_dtype_str, ...) {
  # Create an environment to store the pipeline components
  # pipeline <- new.env(parent = emptyenv())
  devices <- m2d$devices
  unet_dtype <- m2d$unet_dtype
  device_cpu <- m2d$device_cpu
  device_cuda <- m2d$device_cuda
  
  pipeline <- list()
  # Load models into the environment
  if(i2i){
    message("Loading image encoder...")
    pipeline$encoder <- load_model_component(component = "encoder",
                                             model_name,
                                             devices$encoder)
  }
  message("Loading text_encoder...")
  pipeline$text_encoder <- load_model_component("text_encoder", model_name,
                                                devices$text_encoder)
  if(model_name == "sdxl"){
    message("Loading text_encoder2...")
    pipeline$text_encoder2 <- load_model_component(component = "text_encoder2",
                                                   model_name,
                                                   devices$text_encoder2)
  }
  message("Loading unet...")
  pipeline$unet <- load_model_component("unet", model_name,
                               device = devices$unet,
                               unet_dtype_str = unet_dtype_str)

  message("Loading image decoder...")
  pipeline$decoder <- load_model_component("decoder", model_name,
                                           torch::torch_device(devices$decoder))
  
  # Store configuration
  pipeline$devices <- devices
  
  # Add a class for S3 method dispatch if needed
  # class(pipeline) <- c("diffusion_pipeline", "environment")
  
  return(pipeline)
}
