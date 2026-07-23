#' Download and Prepare LTX-2.3 Model Weights
#'
#' Downloads the LTX-2.3 distilled checkpoint (46 GB, LTX-2 Community
#' License) and the Gemma3 text encoder from HuggingFace with an explicit
#' consent prompt, then quantizes the transformer to the local fp8
#' artifact (~26 GB) used by the GPU-poor pipeline.
#'
#' @name download_ltx23
NULL

.ltx23_checkpoint_repo <- "Lightricks/LTX-2.3"
.ltx23_checkpoint_file <- "ltx-2.3-22b-distilled-1.1.safetensors"
.ltx23_text_encoder_repo <- "Lightricks/LTX-2"

.ltx23_disk_free_gb <- function(path) {
    out <- tryCatch(
                    system2("df", c("-Pk", shQuote(path)), stdout = TRUE, stderr = FALSE),
                    error = function(e) NULL
    )
    if (is.null(out) || length(out) < 2L) {
        return(NA_real_)
    }
    fields <- strsplit(trimws(out[[2]]), "\\s+")[[1]]
    as.numeric(fields[4]) / 1024 ^ 2
}

.ltx23_consent <- function(what) {
    if (isTRUE(getOption("diffuseR.consent"))) {
        return(TRUE)
    }
    if (!interactive()) {
        stop(
             "Cannot download models in non-interactive mode without consent. ",
             "Set options(diffuseR.consent = TRUE) to allow downloads.",
             call. = FALSE
        )
    }
    isTRUE(utils::askYesNo(paste0("Download ", what, "?")))
}

#' Download the LTX-2.3 checkpoint and build the fp8 artifact
#'
#' Skips work that is already done: a valid fp8 manifest short-circuits
#' everything; a cached 46 GB source skips the download. The source file
#' may be deleted after quantization (it is never removed automatically).
#'
#' @param quantize Logical. Build the fp8 artifact after downloading.
#' @param output_dir Directory for the fp8 artifact.
#' @param text_encoder Logical. Also fetch the Gemma3 text encoder and
#'   tokenizer (~25 GB, shared with LTX-2.0; from the Lightricks/LTX-2
#'   repo).
#' @param verbose Logical.
#'
#' @return Invisibly, a list with \code{checkpoint} (source path or NULL),
#'   \code{fp8_dir}, and \code{text_encoder_dir}.
#'
#' @export
download_ltx2 <- function(quantize = TRUE,
                          output_dir = file.path(tools::R_user_dir("diffuseR", "data"), "ltx2.3-fp8"),
                          text_encoder = TRUE, verbose = TRUE) {
    if (!requireNamespace("hfhub", quietly = TRUE)) {
        stop("The hfhub package is required to download model weights.")
    }
    result <- list(checkpoint = NULL, fp8_dir = output_dir,
                   text_encoder_dir = NULL)

    manifest_path <- file.path(output_dir, "manifest.json")
    have_fp8 <- file.exists(manifest_path) && {
        m <- jsonlite::fromJSON(manifest_path)
        all(file.exists(file.path(output_dir, m$shards)))
    }

    if (!have_fp8 || !quantize) {
        cached <- tryCatch(
                           hfhub::hub_download(.ltx23_checkpoint_repo, .ltx23_checkpoint_file,
                local_files_only = TRUE),
                           error = function(e) NULL
        )
        if (is.null(cached) && !have_fp8) {
            free <- .ltx23_disk_free_gb(path.expand("~"))
            if (!is.na(free) && free < 75) {
                warning(sprintf(
                                "Only %.0f GB free; the download + fp8 artifact need ~75 GB.", free
                    ))
            }
            ok <- .ltx23_consent(paste0(
                                        "the LTX-2.3 distilled checkpoint (46 GB) plus a ~26 GB local fp8 ",
                                        "artifact from HuggingFace (weights under the LTX-2 Community License)"
                ))
            if (!ok) {
                stop("Download cancelled.", call. = FALSE)
            }
            if (verbose) {
                message("Downloading ", .ltx23_checkpoint_file, " (46 GB)...")
            }
            cached <- hfhub::hub_download(.ltx23_checkpoint_repo, .ltx23_checkpoint_file)
        }
        result$checkpoint <- cached

        if (quantize && !have_fp8) {
            if (verbose) {
                message("Quantizing transformer linears to fp8 (one-time)...")
            }
            ltx23_quantize_fp8(cached, output_dir, verbose = verbose)
            if (verbose) {
                message(
                        "FP8 artifact ready: ", output_dir, "\n",
                        "The 46 GB source in the HuggingFace cache may be deleted if ",
                        "you do not need bf16 weights."
                )
            }
        }
    } else if (verbose) {
        message("FP8 artifact already present: ", output_dir)
    }

    if (text_encoder) {
        te_files <- c(
                      "text_encoder/config.json",
                      sprintf("text_encoder/model-%05d-of-00011.safetensors", 1:11),
                      "text_encoder/model.safetensors.index.json",
                      "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
                      "tokenizer/special_tokens_map.json"
        )
        have_te <- !is.null(tryCatch(
                                     hfhub::hub_download(.ltx23_text_encoder_repo, te_files[[2]],
                    local_files_only = TRUE),
                                     error = function(e) NULL
            ))
        if (!have_te) {
            ok <- .ltx23_consent(
                                 "the Gemma3 text encoder and tokenizer (~25 GB, Lightricks/LTX-2)"
            )
            if (!ok) {
                stop("Download cancelled.", call. = FALSE)
            }
        }
        if (verbose && !have_te) {
            message("Downloading Gemma3 text encoder...")
        }
        paths <- vapply(te_files, function(f) {
            tryCatch(hfhub::hub_download(.ltx23_text_encoder_repo, f),
                     error = function(e) NA_character_)
        }, character(1))
        result$text_encoder_dir <- dirname(paths[[2]])
    }

    invisible(result)
}

# TRUE when every file is already in the local hub cache, i.e. no
# network fetch will happen. Consent gates key on this - never on a
# single sentinel file, and never on artifact presence (a completed
# artifact must not license re-downloading sources).
.hub_all_cached <- function(repo, files, repo_type = NULL) {
    all(vapply(files, function(f) {
        args <- list(repo, f, local_files_only = TRUE)
        if (!is.null(repo_type)) {
            args$repo_type <- repo_type
        }
        !is.null(tryCatch(do.call(hfhub::hub_download, args),
                          error = function(e) NULL))
    }, logical(1)))
}
