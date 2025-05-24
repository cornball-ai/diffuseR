#' models2devices
#' @description This function sets up the model directory, device configuration, and data types for the Stable Diffusion model.
#' It checks the validity of the model name and devices, and downloads the model if necessary.
#' It also sets the data type for the UNet model based on the device configuration.
#' 
#' @param model_name A character string representing the name of the model to be used.
#' @param devices A character string or a named list specifying the devices for different components of the model.
#' @param unet_dtype_str A character string specifying the data type for the UNet model.
#' 
#' @return A list containing the model directory, model files, device configuration, UNet data type, and CPU/CUDA devices.
#' @export
models2devices <- function(model_name, devices = "cpu", unet_dtype_str = NULL) {
  # Check if the model name is valid
  if (is.null(model_name) || !is.character(model_name)) {
    stop("Invalid model name")
  }
  
  # Check if the devices parameter is valid
  if (is.null(devices) || (!is.character(devices) && !is.list(devices))) {
    stop("Invalid devices parameter")
  }
  
  # Check if the unet_dtype_str parameter is valid
  if (!is.null(unet_dtype_str) && !is.character(unet_dtype_str)) {
    stop("Invalid unet_dtype_str parameter")
  }
  # Standardize the 'devices' parameter
  if (is.character(devices) && length(devices) == 1) {
    devices <- list(
      unet = devices,
      decoder = devices,
      text_encoder = devices,
      encoder = devices  # optional, but included by default
    )
  } else if (is.list(devices)) {
    required_keys <- c("unet", "decoder", "text_encoder")
    missing_keys <- setdiff(required_keys, names(devices))
    
    if (length(missing_keys) > 0) {
      stop(paste0(
        "'devices' list must contain: ", paste(required_keys, collapse = ", "), ". ",
        "Optional: 'encoder'. Missing: ", paste(missing_keys, collapse = ", ")
      ))
    }
    
    # Fill in encoder if not present, using decoder's device
    if (!"encoder" %in% names(devices)) {
      if(devices$decoder == "cpu"){
        devices$encoder <- "cpu"
      } else if(devices$decoder == "cuda"){
        devices$encoder <- "cuda"
      } else {
        stop("Invalid device for encoder")
      }
    }
  } else {
    stop(paste0("'devices' must be either a single device string or a named ",
                "list with 'unet', 'decoder', and 'text_encoder'. Optionally: 'encoder'."))
  }
  
  device_cpu <- torch::torch_device("cpu")
  device_cuda <- torch::torch_device("cuda")
  
  if(devices$unet == "cpu"){
    unet_dtype <- torch::torch_float32() # aka torch_half() -> torch_Half 
  } else {
    if(devices$unet == "cuda"){
      if(is.null(unet_dtype_str)){
        unet_dtype <- torch::torch_float16() # aka torch_half() -> torch_Half
      } else {
        if(unet_dtype_str == "float16"){
          unet_dtype <- torch::torch_float16() # aka torch_half() -> torch_Half
        } else if(unet_dtype_str == "float32"){
          unet_dtype <- torch::torch_float32() # aka torch_float() -> torch_Float
        } else {
          stop("Invalid dtype")
        }
      }
    } else {
      stop("Invalid device")
    }
  }
  
  # Check if the model is downloaded
  models <- download_model(model_name, devices, unet_dtype_str)
  model_dir <- models$model_dir
  model_files <- models$model_files
  return(list(model_dir = model_dir,
              model_files = model_files,
              devices = devices,
              unet_dtype = unet_dtype,
              device_cpu = device_cpu,
              device_cuda = device_cuda))
}