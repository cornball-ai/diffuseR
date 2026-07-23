#' Download and Prepare FLUX.2 Klein 4B Weights
#'
#' Downloads FLUX.2-klein-4B from HuggingFace (Apache-2.0, ungated) and
#' quantizes the 4B transformer to a local fp8 (~4 GB) or NF4 (~2.3 GB)
#' artifact.
#'
#' @name download_flux2
NULL

.flux2_repo <- "black-forest-labs/FLUX.2-klein-4B"

.flux2_transformer_files <- c("transformer/config.json",
                              "transformer/diffusion_pytorch_model.safetensors")

.flux2_support_files <- c(
                          "vae/config.json",
                          "vae/diffusion_pytorch_model.safetensors",
                          "text_encoder/config.json",
                          "text_encoder/model.safetensors.index.json",
                          sprintf("text_encoder/model-%05d-of-00002.safetensors", 1:2),
                          "tokenizer/tokenizer.json",
                          "tokenizer/tokenizer_config.json",
                          "tokenizer/special_tokens_map.json",
                          "scheduler/scheduler_config.json"
)

#' Download FLUX.2-klein-4B and build the quantized artifact
#'
#' Skips work already done: a valid quantized manifest short-circuits
#' the transformer download; cached files are not re-fetched. No token
#' is needed (the repo is ungated). The bf16 transformer source
#' (~7.8 GB in the HuggingFace cache) may be deleted after quantization.
#'
#' @param quantize Logical. Build the quantized artifact.
#' @param precision "auto" (default: fp8 when safetensors supports
#'   float8, else nf4), "fp8" (~4 GB, GPU-resident; near-bf16 quality),
#'   or "nf4" (~2.3 GB).
#' @param output_dir Directory for the quantized artifact.
#' @param text_encoders Logical. Also fetch the Qwen3 text encoder,
#'   tokenizer, VAE, and scheduler config (~8.3 GB).
#' @param verbose Logical.
#'
#' @return Invisibly, a list with \code{transformer_dir},
#'   \code{artifact_dir}, and \code{support} (named file paths).
#'
#' @export
download_flux2_klein <- function(quantize = TRUE,
                                 precision = c("auto", "fp8", "nf4"),
                                 output_dir = NULL, text_encoders = TRUE,
                                 verbose = TRUE) {
    precision <- match.arg(precision)
    precision <- .flux_resolve_precision(precision,
        file.path(tools::R_user_dir("diffuseR", "data"), "flux2-klein-4b-"))
    # Explicit fp8 without float8 support: warn + build nf4 rather than fail.
    precision <- .st_graceful_precision(precision, mode = "write")
    if (is.null(output_dir)) {
        output_dir <- file.path(tools::R_user_dir("diffuseR", "data"),
                                paste0("flux2-klein-4b-", precision))
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
                           hfhub::hub_download(.flux2_repo, .flux2_transformer_files[[2]],
                local_files_only = TRUE),
                           error = function(e) NULL
        )
        if (is.null(cached) || !.hub_all_cached(.flux2_repo, .flux2_transformer_files)) {
            free <- .ltx23_disk_free_gb(path.expand("~"))
            if (!is.na(free) && free < 25) {
                warning(sprintf(
                                "Only %.0f GB free; the download + %s artifact need ~25 GB.",
                                free, precision
                    ))
            }
            ok <- .ltx23_consent(paste0(
                                        "FLUX.2-klein-4B: the 7.8 GB bf16 transformer plus a local ",
                                        precision, " artifact (Apache-2.0, ungated)"
                ))
            if (!ok) {
                stop("Download cancelled.", call. = FALSE)
            }
            if (verbose) {
                message("Downloading the FLUX.2-klein-4B transformer (7.8 GB)...")
            }
        }
        paths <- vapply(.flux2_transformer_files, function(f) {
            hfhub::hub_download(.flux2_repo, f)
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
                        "The 7.8 GB source in the HuggingFace cache may be ",
                        "deleted if you do not need bf16 weights."
                )
            }
        }
    } else if (verbose) {
        message(toupper(precision), " artifact already present: ", output_dir)
    }

    if (text_encoders) {
        have_te <- .hub_all_cached(.flux2_repo, .flux2_support_files)
        if (!have_te) {
            ok <- .ltx23_consent(
                                 "the Qwen3-4B text encoder, tokenizer, and VAE (~8.3 GB)"
            )
            if (!ok) {
                stop("Download cancelled.", call. = FALSE)
            }
            if (verbose) {
                message("Downloading text encoder + VAE...")
            }
        }
        result$support <- vapply(.flux2_support_files, function(f) {
            hfhub::hub_download(.flux2_repo, f)
        }, character(1))
    }

    invisible(result)
}
