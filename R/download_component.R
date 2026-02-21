#' Download a single TorchScript model component
#'
#' Downloads a specific model component file (e.g., UNet, decoder, text encoder)
#' using \code{hfhub::hub_download()} from the cornball-ai dataset repos.
#'
#' @param model_name Character string, the name of the model (e.g., \code{"sd21"}).
#' @param component Character string, the component to download (e.g., \code{"unet"}, \code{"decoder"}).
#' @param device Character string, the device type (e.g., \code{"cpu"} or \code{"cuda"}).
#' @param overwrite Logical; if \code{TRUE}, force re-download even if cached.
#' @param show_progress Logical; if \code{TRUE} (default), displays progress during download.
#'
#' @return The local file path to the downloaded component (character string).
#' @export
#'
#' @examples
#' \dontrun{
#' path <- download_component("sd21", "text_encoder", "cpu")
#' }
download_component <- function (model_name = "sd21", component, device = "cpu",
                                overwrite = FALSE, show_progress = TRUE) {
    filename <- paste0(component, "-", device, ".pt")

    if (overwrite) {
        # Force download (bypass cache)
        if (!requireNamespace("hfhub", quietly = TRUE)) {
            stop("Package 'hfhub' is required. Install with: install.packages('hfhub')")
        }
        repo_id <- paste0("cornball-ai/", model_name, "-R")
        return(hfhub::hub_download(repo_id, filename,
                repo_type = "dataset", force_download = TRUE))
    }

    hf_download_pt(model_name, filename, download = TRUE)
}

