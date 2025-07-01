#' Download TorchScript model files for Stable Diffusion
#'
#' Downloads the required model files (e.g., UNet, decoder, text encoder, and tokenizer) 
#' for a given Stable Diffusion model into a local user-specific data directory.
#'
#' The files will be stored in a persistent path returned by [tools::R_user_dir()], typically:
#' \itemize{
#'   \item macOS: `~/Library/Application Support/diffuseR/`
#'   \item Linux: `~/.local/share/R/diffuseR/`
#'   \item Windows: `C:/Users/<username>/AppData/Local/diffuseR/`
#' }
#'
#' Each model is stored in its own subdirectory for better organization.
#' If the files already exist, they will not be downloaded again unless `overwrite = TRUE`.
#' @param model_name Name of the model (e.g., "sd21" for stable-diffusion-2-1)
#' @param devices Either a single device string or a named list with elements 'unet', 'decoder', 'text_encoder'; optionally 'encoder'
#' @param unet_dtype_str Optional: "float16" or "float32" (only applies if unet uses CUDA)
#' @param overwrite If TRUE, overwrite existing model files
#' @param show_progress Show download progress messages
#' @param download_models If TRUE, download the model files from Hugging Face
#'
#' @return A list with `model_dir` and `model_files`
#' @export
#``
#' @examples
#' \dontrun{
#' model_dir <- download_model("sd21")
#' }
#'
download_model <- function(model_name = "sd21",
                           devices = list(unet = "cpu", decoder = "cpu", text_encoder = "cpu"),
                           unet_dtype_str = NULL,
                           overwrite = FALSE,
                           show_progress = TRUE,
                           download_models = FALSE) {
  # Normalize 'devices'
  if (is.character(devices) && length(devices) == 1) {
    # For SDXL, we need to handle both text encoders
    if (model_name == "sdxl") {
      devices <- list(
        unet = devices,
        decoder = devices,
        text_encoder1 = devices,
        text_encoder2 = devices,
        encoder = devices
      )
    } else {
      devices <- list(
        unet = devices,
        decoder = devices,
        text_encoder = devices,
        encoder = devices
      )
    }
  } else if (is.list(devices)) {
    # Define required keys based on model
    if (model_name == "sdxl") {
      required_keys <- c("unet", "decoder", "text_encoder", "text_encoder2")
      # Handle backward compatibility - if only 'text_encoder' is provided, use it for both
      if ("text_encoder" %in% names(devices) && 
          !("text_encoder2" %in% names(devices))) {
        devices$text_encoder2 <- devices$text_encoder
      }
    } else {
      required_keys <- c("unet", "decoder", "text_encoder")
    }
    
    missing_keys <- setdiff(required_keys, names(devices))
    if (length(missing_keys) > 0) {
      stop(paste0("Missing required devices: ", paste(missing_keys, collapse = ", ")))
    }
    
    if (!"encoder" %in% names(devices)) {
      devices$encoder <- devices$decoder
    }
  } else {
    stop("'devices' must be a device string or a named list with required components")
  }
  
  # Set up model storage path
  base_dir <- tools::R_user_dir("diffuseR", "data")
  model_dir <- file.path(base_dir, model_name)
  dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Assemble model files
  if(model_name == "sdxl"){
    model_names <- c("unet", "decoder", "text_encoder", "text_encoder2", "encoder")
  } else if(model_name == "sd21"){
    model_names <- c("unet", "decoder", "text_encoder", "encoder")
  } else {
    stop("Unsupported model name: ", model_name)
  }
  
  model_files <- character(length(model_names))
  
  for (i in seq_along(model_names)) {
    name <- model_names[i]
    device <- devices[[name]]
    
    if (is.null(device)) {
      stop(paste0("Device not specified for component: ", name))
    }
    
    if (name == "unet" && device != "cpu") {
      if (is.null(unet_dtype_str)) {
        dtype <- "float16"
      } else {
        dtype <- unet_dtype_str
      }
      if (!dtype %in% c("float16", "float32")) {
        stop("Invalid unet_dtype_str: must be 'float16' or 'float32'")
      }
      model_files[i] <- paste0("unet-cuda-", dtype, ".pt")
    } else {
      model_files[i] <- paste0(name, "-", device, ".pt")
    }
  }
  
  # Download files
  if (download_models) {
    repo_url <- paste0("https://huggingface.co/datasets/cornball-ai/", model_name, "-R/resolve/main/")
    for (file in model_files) {
      dest_path <- file.path(model_dir, file)
      if (!file.exists(dest_path) || overwrite) {
        url <- paste0(repo_url, file)
        message("Downloading ", file, "...")
        tryCatch({
          utils::download.file(
            url = url,
            destfile = dest_path,
            mode = "wb",
            quiet = !show_progress
          )
        }, error = function(e) {
          warning("Download failed: ", file, " - ", e$message)
          if (file.exists(dest_path)) unlink(dest_path)
        })
      } else {
        message("File already exists: ", file)
      }
    }
  } else {
      message("Skipping model download (download_models == FALSE)")
  }
  # Check for missing
  missing_files <- model_files[!file.exists(file.path(model_dir, model_files))]
  if (length(missing_files) > 0) {
    stop("Missing model files: ", paste(missing_files, collapse = ", "))
  }
  
  list(model_dir = model_dir,
       model_files = model_files,
       model_names = model_names)
}
