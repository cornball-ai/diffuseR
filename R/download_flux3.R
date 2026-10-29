#' Download and Prepare FLUX 3 Image Klein Weights
#'
#' Downloads FLUX 3 Image Klein from HuggingFace (gated; FLUX Kommunity
#' License) and quantizes the 5.6B transformer to a local fp8 (~6 GB) or
#' NF4 artifact. Only the diffusers-format files are fetched: the bf16
#' transformer shards, the Qwen3-VL text encoder, the VAE and the
#' tokenizer. BFL's own root \code{model.safetensors} and the
#' \code{variants/} quantizations are not used.
#'
#' @name download_flux3
NULL

.flux3_transformer_files <- c(
                              "transformer/config.json",
                              "transformer/diffusion_pytorch_model.safetensors.index.json",
                              sprintf("transformer/diffusion_pytorch_model-%05d-of-00002.safetensors",
                                      1:2)
)

.flux3_support_files <- c(
                          "LICENSE.md",
                          "NOTICE.md",
                          "vae/config.json",
                          "vae/diffusion_pytorch_model.safetensors",
                          "text_encoder/config.json",
                          "text_encoder/model.safetensors.index.json",
                          sprintf("text_encoder/model-%05d-of-00002.safetensors", 1:2),
                          "text_encoder/tokenizer.json",
                          "text_encoder/tokenizer_config.json",
                          "text_encoder/chat_template.json"
)

# hub_download pinned to the validated snapshot, with the gated-repo
# 401/403 turned into an actionable error
.flux3_download <- function(file, revision = .flux3_revision) {
    tryCatch(
             hfhub::hub_download(.flux3_repo, file, revision = revision),
             error = function(e) {
        msg <- conditionMessage(e)
        if (grepl("401|403|[Uu]nauthorized|[Ff]orbidden", msg)) {
            stop(
                 "FLUX 3 Image Klein is a gated HuggingFace repo under the ",
                 "FLUX Kommunity License. To download:\n",
                 "  1. Log in at https://huggingface.co/", .flux3_repo,
                 " and accept the license.\n",
                 "  2. Create a read token at https://huggingface.co/settings/tokens\n",
                 "  3. Sys.setenv(HF_TOKEN = \"hf_...\") and retry.",
                 call. = FALSE
            )
        }
        stop(e)
    }
    )
}

#' Download FLUX 3 Image Klein and build the quantized artifact
#'
#' Skips work already done: a valid quantized manifest short-circuits
#' the transformer download; cached files are not re-fetched. Needs
#' \code{HF_TOKEN} for an account that accepted the license. The bf16
#' transformer source (~11 GB in the HuggingFace cache) may be deleted
#' after quantization.
#'
#' The FLUX Kommunity License permits non-commercial use, and commercial
#' use of outputs by Qualifying Users (under US$5M gross annualized
#' revenue) who filter or review outputs before distributing them and
#' disclose AI generation where the law requires it. Read LICENSE.md in
#' the downloaded snapshot for the full terms. The quantized artifact is
#' a Derivative; keep it local.
#'
#' @param quantize Logical. Build the quantized artifact.
#' @param precision "auto" (default: fp8 when safetensors supports
#'   float8, else nf4), "fp8" (~6 GB), or "nf4".
#' @param output_dir Directory for the quantized artifact.
#' @param text_encoders Logical. Also fetch the Qwen3-VL text encoder,
#'   tokenizer and VAE (~9.2 GB).
#' @param text_precision Quantize the Qwen3-VL text model too (with
#'   \code{quantize}): "fp8" (default; a 4.4 GB artifact, host memory at
#'   1024x1024 20.2 -> 15.5 GB, images close to bf16 text: 27 dB PSNR),
#'   "nf4" (2.8 GB, 13.5 GB host, but a visibly different image for the
#'   same seed), or "bf16" (keep only the cached source).
#'   \code{\link{flux3_load_pipeline}} picks the artifact up.
#' @param verbose Logical.
#'
#' @return Invisibly, a list with \code{transformer_dir},
#'   \code{artifact_dir}, \code{support} (named file paths), and
#'   \code{text_artifact_dir} when the text model was quantized.
#'
#' @export
download_flux3_klein <- function(quantize = TRUE,
                                 precision = c("auto", "fp8", "nf4"),
                                 output_dir = NULL, text_encoders = TRUE,
                                 text_precision = c("fp8", "nf4", "bf16"),
                                 verbose = TRUE) {
    precision <- match.arg(precision)
    text_precision <- match.arg(text_precision)
    precision <- .flux_resolve_precision(precision, .flux3_artifact_prefix())
    precision <- .st_graceful_precision(precision, mode = "write")
    output_dir <- output_dir %||% paste0(.flux3_artifact_prefix(), precision)
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
        if (!.hub_all_cached(.flux3_repo, .flux3_transformer_files)) {
            free <- .ltx23_disk_free_gb(path.expand("~"))
            if (!is.na(free) && free < 40) {
                warning(sprintf(
                                "Only %.0f GB free; the downloads + %s artifact need ~30 GB.",
                                free, precision))
            }
            ok <- .ltx23_consent(paste0(
                                        "FLUX 3 Image Klein: the 11 GB bf16 transformer plus a local ",
                                        precision, " artifact (FLUX Kommunity License, gated)"))
            if (!ok) {
                stop("Download cancelled.", call. = FALSE)
            }
            if (verbose) {
                message("Downloading the FLUX 3 Image Klein transformer (11 GB)...")
            }
        }
        paths <- vapply(.flux3_transformer_files, .flux3_download, character(1))
        result$transformer_dir <- dirname(paths[[1]])

        if (quantize && !have_artifact) {
            if (verbose) {
                message("Quantizing transformer linears to ", precision,
                        " (one-time)...")
            }
            flux_quantize(result$transformer_dir, output_dir,
                          format = precision, verbose = verbose)
            if (verbose) {
                message(toupper(precision), " artifact ready: ", output_dir,
                        "\nThe 11 GB source in the HuggingFace cache may be ",
                        "deleted if you do not need bf16 weights.")
            }
        }
    } else if (verbose) {
        message(toupper(precision), " artifact already present: ", output_dir)
    }

    if (text_encoders) {
        if (!.hub_all_cached(.flux3_repo, .flux3_support_files)) {
            ok <- .ltx23_consent(
                                 "the Qwen3-VL-4B text encoder, tokenizer and VAE (~9.2 GB)")
            if (!ok) {
                stop("Download cancelled.", call. = FALSE)
            }
            if (verbose) {
                message("Downloading text encoder + VAE...")
            }
        }
        result$support <- vapply(.flux3_support_files, .flux3_download,
                                 character(1))
        if (quantize && text_precision != "bf16") {
            text_precision <- .st_graceful_precision(text_precision,
                                                     mode = "write")
            result$text_artifact_dir <- .flux3_text_artifact(text_precision)
            qwen3_quantize(dirname(result$support[["text_encoder/config.json"]]),
                           result$text_artifact_dir, format = text_precision,
                           verbose = verbose)
        }
    }

    invisible(result)
}
