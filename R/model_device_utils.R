# model_device_utils.R

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
      devices$encoder <- NULL
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