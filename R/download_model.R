#' Download a single TorchScript component via hfhub
#'
#' Internal helper that wraps \code{hfhub::hub_download()} with the cornball-ai
#' dataset repo naming convention. Falls back to legacy \code{R_user_dir()} paths
#' if the file exists there but not yet in the hfhub cache.
#'
#' @param model_name Model name (e.g., \code{"sd21"}, \code{"sdxl"}).
#' @param filename Filename within the repo (e.g., \code{"unet-cpu.pt"}).
#' @param download If \code{TRUE}, download from HuggingFace when not cached.
#'
#' @return The local file path (character string).
#' @keywords internal
hf_download_pt <- function (model_name, filename, download = TRUE) {
    if (!requireNamespace("hfhub", quietly = TRUE)) {
        stop("Package 'hfhub' is required. Install with: install.packages('hfhub')")
    }

    repo_id <- paste0("cornball-ai/", model_name, "-R")

    # 1. Try hfhub cache first (no network)
    path <- tryCatch(
        hfhub::hub_download(repo_id, filename,
            repo_type = "dataset", local_files_only = TRUE),
        error = function (e) NULL
    )
    if (!is.null(path) && file.exists(path)) return(path)

    # 2. Check legacy R_user_dir location
    legacy_path <- file.path(tools::R_user_dir("diffuseR", "data"),
        model_name, filename)
    if (file.exists(legacy_path)) return(legacy_path)

    # 3. Download via hfhub
    if (!download) {
        stop("Component '", filename, "' for model '", model_name,
            "' not found locally. Set download=TRUE to download.")
    }

    hfhub::hub_download(repo_id, filename, repo_type = "dataset")
}

#' Download TorchScript model files for Stable Diffusion
#'
#' Downloads the required model files (e.g., UNet, decoder, text encoder)
#' for a given Stable Diffusion model using \code{hfhub::hub_download()}.
#'
#' Files are cached by hfhub (typically \code{~/.cache/huggingface/hub/}).
#' Legacy files in the old \code{R_user_dir()} location are also recognized.
#'
#' @param model_name Name of the model (e.g., "sd21" for stable-diffusion-2-1)
#' @param devices Either a single device string or a named list with elements 'unet', 'decoder', 'text_encoder'; optionally 'encoder'
#' @param unet_dtype_str Optional: "float16" or "float32" (only applies if unet uses CUDA)
#' @param overwrite If TRUE, force re-download of model files
#' @param show_progress Show download progress messages
#' @param download_models If TRUE, download the model files from HuggingFace
#'
#' @return A named list of full file paths, keyed by component name.
#' @export
#'
#' @examples
#' \dontrun{
#' paths <- download_model("sd21")
#' }
#'
download_model <- function(
    model_name = "sd21",
    devices = list(unet = "cpu", decoder = "cpu", text_encoder = "cpu"),
    unet_dtype_str = NULL,
    overwrite = FALSE,
    show_progress = TRUE,
    download_models = FALSE
) {
    # Normalize 'devices'
    if (is.character(devices) && length(devices) == 1) {
        if (model_name == "sdxl") {
            devices <- list(
                unet = devices,
                decoder = devices,
                text_encoder = devices,
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
        if (model_name == "sdxl") {
            required_keys <- c("unet", "decoder", "text_encoder", "text_encoder2")
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

    # Determine component names and filenames
    if (model_name == "sdxl") {
        model_names <- c("unet", "decoder", "text_encoder", "text_encoder2", "encoder")
    } else if (model_name == "sd21") {
        model_names <- c("unet", "decoder", "text_encoder", "encoder")
    } else {
        stop("Unsupported model name: ", model_name)
    }

    filenames <- character(length(model_names))
    for (i in seq_along(model_names)) {
        name <- model_names[i]
        device <- devices[[name]]
        if (is.null(device)) {
            stop("Device not specified for component: ", name)
        }
        if (name == "unet" && device != "cpu") {
            dtype <- if (is.null(unet_dtype_str)) "float16" else unet_dtype_str
            if (!dtype %in% c("float16", "float32")) {
                stop("Invalid unet_dtype_str: must be 'float16' or 'float32'")
            }
            filenames[i] <- paste0("unet-cuda-", dtype, ".pt")
        } else {
            filenames[i] <- paste0(name, "-", device, ".pt")
        }
    }

    # Check which files are already available (hfhub cache or legacy)
    already_available <- vapply(filenames, function(f) {
            # Check hfhub cache
            path <- tryCatch(
                hfhub::hub_download(paste0("cornball-ai/", model_name, "-R"), f,
                    repo_type = "dataset", local_files_only = TRUE),
                error = function(e) NULL
            )
            if (!is.null(path) && file.exists(path)) return(TRUE)
            # Check legacy location
            legacy <- file.path(tools::R_user_dir("diffuseR", "data"), model_name, f)
            file.exists(legacy)
        }, logical(1))

    needs_download <- !already_available | overwrite

    # Interactive consent when files actually need downloading
    if (download_models && any(needs_download)) {
        if (interactive()) {
            ans <- utils::askYesNo(
                paste0("Download '", model_name, "' model files from HuggingFace?"),
                default = TRUE
            )
            if (!isTRUE(ans)) {
                stop("Download cancelled.", call. = FALSE)
            }
        }
    }

    # Resolve all paths (download if needed and allowed)
    model_paths <- character(length(model_names))
    names(model_paths) <- model_names
    for (i in seq_along(model_names)) {
        if (needs_download[i] && !download_models) {
            # Not available and not allowed to download — will error
            model_paths[i] <- tryCatch(
                hf_download_pt(model_name, filenames[i], download = FALSE),
                error = function(e) NA_character_
            )
        } else {
            model_paths[i] <- tryCatch(
                hf_download_pt(model_name, filenames[i], download = download_models),
                error = function(e) {
                    warning("Download failed for ", filenames[i], ": ", e$message)
                    NA_character_
                }
            )
        }
    }

    missing <- model_names[is.na(model_paths)]
    if (length(missing) > 0) {
        stop("Missing model files: ", paste(filenames[is.na(model_paths)], collapse = ", "))
    }

    model_paths
}

