#' Download and Prepare Z-Image-Turbo Weights
#'
#' Downloads Z-Image-Turbo from HuggingFace (Apache-2.0, ungated) and
#' quantizes the 6B transformer to a local fp8 (~6.3 GB) or NF4
#' (~3.6 GB) artifact. The checkpoint ships the transformer in float32
#' (24.6 GB), so the one-time quantize saves a lot of disk and load
#' time.
#'
#' @name download_zimage
NULL

.zimage_repo <- "Tongyi-MAI/Z-Image-Turbo"

.zimage_transformer_files <- c(
                               "transformer/config.json",
                               "transformer/diffusion_pytorch_model.safetensors.index.json",
                               sprintf("transformer/diffusion_pytorch_model-%05d-of-00003.safetensors",
                                       1:3)
)

.zimage_support_files <- c(
                           "vae/config.json",
                           "vae/diffusion_pytorch_model.safetensors",
                           "text_encoder/config.json",
                           "text_encoder/model.safetensors.index.json",
                           sprintf("text_encoder/model-%05d-of-00003.safetensors", 1:3),
                           "tokenizer/tokenizer.json",
                           "tokenizer/tokenizer_config.json",
                           "scheduler/scheduler_config.json"
)

#' Download Z-Image-Turbo and build the quantized artifact
#'
#' Skips work already done: a valid quantized manifest short-circuits
#' the transformer download; cached files are not re-fetched. No token
#' is needed (the repo is ungated). The float32 transformer source
#' (~24.6 GB in the HuggingFace cache) may be deleted after
#' quantization.
#'
#' @param quantize Logical. Build the quantized artifact.
#' @param precision "auto" (default: fp8 when safetensors supports
#'   float8, else nf4), "fp8" (~6.3 GB, GPU-resident; near-bf16
#'   quality), or "nf4" (~3.6 GB).
#' @param output_dir Directory for the quantized artifact.
#' @param text_encoders Logical. Also fetch the Qwen3-4B text encoder,
#'   tokenizer, VAE, and scheduler config (~8.2 GB).
#' @param verbose Logical.
#'
#' @return Invisibly, a list with \code{transformer_dir},
#'   \code{artifact_dir}, and \code{support} (named file paths).
#'
#' @export
download_zimage_turbo <- function(quantize = TRUE,
                                  precision = c("auto", "fp8", "nf4"),
                                  output_dir = NULL, text_encoders = TRUE,
                                  verbose = TRUE) {
    precision <- match.arg(precision)
    precision <- .flux_resolve_precision(precision,
        file.path(tools::R_user_dir("diffuseR", "data"), "zimage-turbo-"))
    # Explicit fp8 without float8 support: warn + build nf4 rather than fail.
    precision <- .st_graceful_precision(precision, mode = "write")
    if (is.null(output_dir)) {
        output_dir <- file.path(tools::R_user_dir("diffuseR", "data"),
                                paste0("zimage-turbo-", precision))
    }
    if (!requireNamespace("hfhub", quietly = TRUE)) {
        stop("The hfhub package is required to download model weights.")
    }
    result <- list(transformer_dir = NULL, artifact_dir = output_dir,
                   support = character(0))

    manifest_path <- file.path(output_dir, "manifest.json")
    have_artifact <- file.exists(manifest_path) && {
        m <- jsonlite::fromJSON(manifest_path)
        all(file.exists(file.path(output_dir, m$shards)))
    }

    if (!have_artifact || !quantize) {
        cached <- tryCatch(
                           hfhub::hub_download(.zimage_repo, .zimage_transformer_files[[3]],
                local_files_only = TRUE),
                           error = function(e) NULL
        )
        if (is.null(cached) || !.hub_all_cached(.zimage_repo, .zimage_transformer_files)) {
            free <- .ltx23_disk_free_gb(path.expand("~"))
            if (!is.na(free) && free < 35) {
                warning(sprintf(
                                "Only %.0f GB free; the download + %s artifact need ~35 GB.",
                                free, precision
                    ))
            }
            ok <- .ltx23_consent(paste0(
                                        "Z-Image-Turbo: the 24.6 GB float32 transformer plus a local ",
                                        precision, " artifact (Apache-2.0, ungated)"
                ))
            if (!ok) {
                stop("Download cancelled.", call. = FALSE)
            }
            if (verbose) {
                message("Downloading the Z-Image-Turbo transformer (24.6 GB)...")
            }
        }
        paths <- vapply(.zimage_transformer_files, function(f) {
            hfhub::hub_download(.zimage_repo, f)
        }, character(1))
        result$transformer_dir <- dirname(paths[[1]])

        if (quantize && !have_artifact) {
            if (verbose) {
                message("Quantizing transformer linears to ", precision,
                        " (one-time)...")
            }
            flux_quantize(result$transformer_dir, output_dir,
                          format = precision, verbose = verbose)
            if (verbose) {
                message(
                        toupper(precision), " artifact ready: ", output_dir, "\n",
                        "The 24.6 GB float32 source in the HuggingFace cache ",
                        "may be deleted if you do not need it."
                )
            }
        }
    } else if (verbose) {
        message(toupper(precision), " artifact already present: ", output_dir)
    }

    if (text_encoders) {
        have_te <- .hub_all_cached(.zimage_repo, .zimage_support_files)
        if (!have_te) {
            ok <- .ltx23_consent(
                                 "the Qwen3-4B text encoder, tokenizer, and VAE (~8.2 GB)"
            )
            if (!ok) {
                stop("Download cancelled.", call. = FALSE)
            }
            if (verbose) {
                message("Downloading text encoder + VAE...")
            }
        }
        result$support <- vapply(.zimage_support_files, function(f) {
            hfhub::hub_download(.zimage_repo, f)
        }, character(1))
    }

    invisible(result)
}
