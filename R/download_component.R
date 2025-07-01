#' Download TorchScript model component for Stable Diffusion
#'
#' Downloads the required model file (e.g., UNet, decoder, text encoder, and tokenizer) 
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
#'
#' @param model_name Character string, the name of the model to download (e.g., `"sd21"`).
#' @param component Character string, the specific model component to download (e.g., `"unet"`, `"decoder"`, `"text_encoder"`).
#' @param device Character string, the device type for which the model is intended (e.g., `"cpu"` or `"cuda"`).
#' @param overwrite Logical; if `TRUE`, re-downloads the model files even if they already exist.
#' @param show_progress Logical; if `TRUE` (default), displays a progress bar during download.
#'
#' @return The local file path to the specific model directory (as a string).
#' @export
#'
#' @examples
#' \dontrun{
#' model_dir <- download_model("sd21")
#' }
download_component <- function(model_name = "sd21",
                               component,
                               device = "cpu",
                               overwrite = FALSE,
                               show_progress = TRUE) {
  # Use "data" instead of "cache" for persistent storage
  base_dir <- tools::R_user_dir("diffuseR", "data")
  
  # Create model-specific subdirectory
  model_dir <- file.path(base_dir, model_name)
  dir.create(model_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Define all required model files
  model_file <- paste0(model_name, "-", device, ".pt")
  
  # Define the remote source
  repo_url <- paste0("https://huggingface.co/datasets/cornball-ai/", model_name, "-R/resolve/main/")
  
  # Download file
  dest_path <- file.path(model_dir, component)

  # Check if file exists and should be overwritten
  if (!file.exists(dest_path) || overwrite) {
    url <- paste0(repo_url, model_file)
    
    message("Downloading ", model_file, " for ", model_name, "...")
    
    # Handle possible download errors
    tryCatch({
      utils::download.file(
        url = url, 
        destfile = dest_path, 
        mode = "wb",
        quiet = !show_progress
      )
    }, error = function(e) {
      warning("Failed to download ", model_file, ": ", e$message)
      # If file was partially downloaded, remove it
      if (file.exists(dest_path)) {
        unlink(dest_path)
      }
    })
  }
  
  # Check if all required files were downloaded successfully
  missing_files <- model_file[!file.exists(file.path(model_dir, model_file))]
  if (length(missing_files) > 0) {
    warning("Model file could not be downloaded: ", 
            paste(missing_files, collapse = ", "))
  }
  
  list(model_dir = model_dir, model_file = model_file)
}
