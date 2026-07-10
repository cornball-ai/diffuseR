#' Download and Prepare FLUX.1-schnell Weights
#'
#' Downloads FLUX.1-schnell from HuggingFace (weights Apache-2.0, but
#' the repo is gated behind a license click-through) and quantizes the
#' 12B transformer to a local NF4 (~7 GB) or fp8 (~12 GB) artifact.
#'
#' @name download_flux
NULL

.flux1_repo <- "black-forest-labs/FLUX.1-schnell"

.flux1_transformer_files <- c(
                              "transformer/config.json",
                              "transformer/diffusion_pytorch_model.safetensors.index.json",
                              sprintf("transformer/diffusion_pytorch_model-%05d-of-00003.safetensors",
                                      1:3)
)

.flux1_support_files <- c("vae/config.json",
                          "vae/diffusion_pytorch_model.safetensors",
                          "text_encoder/config.json",
                          "text_encoder/model.safetensors",
                          "scheduler/scheduler_config.json")

.flux1_t5_files <- c(
                     "text_encoder_2/config.json",
                     "text_encoder_2/model.safetensors.index.json",
                     sprintf("text_encoder_2/model-%05d-of-00002.safetensors", 1:2),
                     "tokenizer_2/tokenizer.json",
                     "tokenizer_2/tokenizer_config.json",
                     "tokenizer_2/special_tokens_map.json"
)

# hub_download with the gated-repo 401/403 turned into an actionable error
.flux1_download <- function(file, ...) {
    tryCatch(
             hfhub::hub_download(.flux1_repo, file, ...),
             error = function(e) {
        msg <- conditionMessage(e)
        if (grepl("401|403|[Uu]nauthorized|[Ff]orbidden", msg)) {
            stop(
                 "FLUX.1-schnell is a gated HuggingFace repo (the weights are ",
                 "Apache-2.0; the gate is a license click-through). To download:\n",
                 "  1. Log in at https://huggingface.co/black-forest-labs/FLUX.1-schnell ",
                 "and accept the license.\n",
                 "  2. Create a read token at https://huggingface.co/settings/tokens\n",
                 "  3. Sys.setenv(HF_TOKEN = \"hf_...\") and retry.",
                 call. = FALSE
            )
        }
        stop(e)
    }
    )
}

#' Download FLUX.1-schnell and build the quantized artifact
#'
#' Skips work already done: a valid quantized manifest short-circuits
#' the transformer download; cached files are not re-fetched. Needs
#' \code{HF_TOKEN} set for the gated repo (see the error message it
#' raises without one). The bf16 transformer source (~24 GB in the
#' HuggingFace cache) may be deleted after quantization.
#'
#' @param quantize Logical. Build the quantized artifact after
#'   downloading.
#' @param precision "nf4" (~7 GB, GPU-resident on 16 GB cards) or
#'   "fp8" (~12 GB, CPU-resident, streamed; near-bf16 quality).
#' @param output_dir Directory for the quantized artifact.
#' @param text_encoders Logical. Also fetch the CLIP + T5 text encoders,
#'   tokenizer, VAE, and scheduler config (~10 GB).
#' @param verbose Logical.
#'
#' @return Invisibly, a list with \code{transformer_dir},
#'   \code{artifact_dir}, and \code{support} (named file paths).
#'
#' @export
download_flux1 <- function(quantize = TRUE, precision = c("nf4", "fp8"),
                           output_dir = NULL, text_encoders = TRUE,
                           verbose = TRUE) {
    precision <- match.arg(precision)
    # Explicit fp8 without float8 support: warn + build nf4 instead of
    # failing in flux_quantize.
    precision <- .st_graceful_precision(precision, mode = "write")
    if (is.null(output_dir)) {
        output_dir <- file.path(tools::R_user_dir("diffuseR", "data"),
                                paste0("flux1-schnell-", precision))
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
                           hfhub::hub_download(.flux1_repo, .flux1_transformer_files[[3]],
                local_files_only = TRUE),
                           error = function(e) NULL
        )
        if (is.null(cached) && !have_artifact) {
            free <- .ltx23_disk_free_gb(path.expand("~"))
            if (!is.na(free) && free < 45) {
                warning(sprintf(
                                "Only %.0f GB free; the download + %s artifact need ~45 GB.",
                                free, precision
                    ))
            }
            ok <- .ltx23_consent(paste0(
                                        "FLUX.1-schnell: the 24 GB bf16 transformer plus a ~",
                    if (precision == "nf4") "7" else "12",
                                        " GB local ", precision,
                                        " artifact (weights Apache-2.0, gated HuggingFace repo)"
                ))
            if (!ok) {
                stop("Download cancelled.", call. = FALSE)
            }
            if (verbose) {
                message("Downloading the FLUX.1-schnell transformer (24 GB)...")
            }
        }
        paths <- vapply(.flux1_transformer_files, .flux1_download, character(1))
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
                        "The 24 GB source in the HuggingFace cache may be deleted ",
                        "if you do not need bf16 weights."
                )
            }
        }
    } else if (verbose) {
        message(toupper(precision), " artifact already present: ", output_dir)
    }

    if (text_encoders) {
        files <- c(.flux1_support_files, .flux1_t5_files)
        have_t5 <- !is.null(tryCatch(
                                     hfhub::hub_download(.flux1_repo, .flux1_t5_files[[3]],
                    local_files_only = TRUE),
                                     error = function(e) NULL
            ))
        if (!have_t5) {
            ok <- .ltx23_consent(
                                 "the FLUX text encoders, VAE, and tokenizer (~10 GB)"
            )
            if (!ok) {
                stop("Download cancelled.", call. = FALSE)
            }
            if (verbose) {
                message("Downloading text encoders + VAE...")
            }
        }
        result$support <- vapply(files, .flux1_download, character(1))
    }

    invisible(result)
}
