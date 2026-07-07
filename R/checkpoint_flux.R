#' FLUX Checkpoint Readers
#'
#' FLUX transformers ship in the diffusers layout: a directory with
#' \code{config.json}, one or more \code{diffusion_pytorch_model*.safetensors}
#' shards, and (when sharded) a
#' \code{diffusion_pytorch_model.safetensors.index.json} weight map.
#' These helpers open that layout behind the same checkpoint interface as
#' \code{\link{ltx23_open_checkpoint}}, so the LTX group loaders and
#' quantization machinery work unchanged. FLUX module names mirror the
#' checkpoint keys 1:1 - no key mapping is needed.
#'
#' @name checkpoint_flux
NULL

# Quantization cast set: every large linear weight in the FLUX blocks,
# including the adaLN modulation linears (norm*.linear are 3.2B params
# across the model; leaving them bf16 would not fit resident on 16 GB).
# Everything else (embedders, q/k norms, final norm, biases) stays bf16.
# Full-size census: 19 double blocks x 14 + 38 single blocks x 6 = 494
# cast weights, ~11.8B of the 12B parameters.
.flux_quant_cast_pattern <- paste0(
                                   "^(",
                                   "transformer_blocks\\.[0-9]+\\.(",
                                   "attn\\.(to_q|to_k|to_v|add_q_proj|add_k_proj|add_v_proj|to_out\\.0|to_add_out)",
                                   "|ff\\.net\\.(0\\.proj|2)|ff_context\\.net\\.(0\\.proj|2)",
                                   "|norm1\\.linear|norm1_context\\.linear",
                                   ")",
                                   "|single_transformer_blocks\\.[0-9]+\\.(",
                                   "attn\\.(to_q|to_k|to_v)|proj_mlp|proj_out|norm\\.linear",
                                   ")",
                                   ")\\.weight$"
)

#' Test whether a FLUX key is in the quantization cast set
#'
#' @param key Character vector of parameter names (diffusers-style).
#'
#' @return Logical vector.
#'
#' @export
flux_is_quant_key <- function(key) {
    grepl(.flux_quant_cast_pattern, key)
}

#' Open a FLUX transformer checkpoint directory
#'
#' Opens a diffusers-layout transformer directory lazily (headers only).
#' Sharded checkpoints are resolved through the index.json weight map;
#' single-file checkpoints are opened directly. The transformer
#' \code{config.json} is attached as \code{$config}.
#'
#' @param transformer_dir Directory containing \code{config.json} and the
#'   \code{diffusion_pytorch_model*.safetensors} file(s).
#'
#' @return An object of class \code{ltx23_checkpoint} (shared checkpoint
#'   interface): list with \code{handle$get_tensor}, \code{keys},
#'   \code{config}, and \code{path}.
#'
#' @export
flux_open_checkpoint <- function(transformer_dir) {
    if (!requireNamespace("safetensors", quietly = TRUE)) {
        stop("The safetensors package is required to read FLUX checkpoints.")
    }
    transformer_dir <- path.expand(transformer_dir)
    if (!dir.exists(transformer_dir)) {
        stop("Checkpoint directory not found: ", transformer_dir)
    }

    config <- NULL
    config_path <- file.path(transformer_dir, "config.json")
    if (file.exists(config_path)) {
        config <- jsonlite::fromJSON(config_path, simplifyVector = TRUE)
    }

    index_path <- file.path(
                            transformer_dir,
                            "diffusion_pytorch_model.safetensors.index.json"
    )
    if (file.exists(index_path)) {
        index <- jsonlite::fromJSON(index_path, simplifyVector = TRUE)
        weight_map <- unlist(index$weight_map)
        shard_files <- unique(weight_map)
        missing <- shard_files[!file.exists(file.path(transformer_dir, shard_files))]
        if (length(missing)) {
            stop("Missing checkpoint shards: ", paste(missing, collapse = ", "))
        }
        handles <- lapply(file.path(transformer_dir, shard_files), function(p) {
            safetensors::safetensors$new(p, framework = "torch")
        })
        names(handles) <- shard_files
        keys <- names(weight_map)
        handle <- list(
                       get_tensor = function(key) {
            shard <- weight_map[[key]]
            if (is.null(shard) || is.na(shard)) {
                stop("Key not found in checkpoint index: ", key)
            }
            handles[[shard]]$get_tensor(key)
        }
        )
    } else {
        single_path <- file.path(transformer_dir,
                                 "diffusion_pytorch_model.safetensors")
        if (!file.exists(single_path)) {
            stop("No diffusion_pytorch_model safetensors (or index) in ",
                 transformer_dir)
        }
        h <- safetensors::safetensors$new(single_path, framework = "torch")
        keys <- setdiff(h$keys(), "__metadata__")
        handle <- list(get_tensor = function(key) h$get_tensor(key))
    }

    structure(
              list(
                   handle = handle,
                   keys = keys,
                   version = NULL,
                   config = config,
                   path = transformer_dir
        ),
              class = "ltx23_checkpoint"
    )
}

#' Open a quantized FLUX artifact directory
#'
#' Opens the sharded NF4/fp8 artifact written by
#' \code{\link{flux_quantize}} through the shared checkpoint interface.
#' The manifest's embedded transformer config and \code{format} ride
#' along, so \code{\link{flux_load_transformer}} needs nothing else.
#'
#' @param dir The quantized artifact directory (with manifest.json).
#'
#' @return An \code{ltx23_checkpoint} with \code{$format} set.
#'
#' @export
flux_open_quantized <- function(dir) {
    manifest_path <- file.path(dir, "manifest.json")
    if (!file.exists(manifest_path)) {
        stop("No manifest.json in ", dir, "; run flux_quantize() first.")
    }
    ltx23_open_fp8_checkpoint(dir)
}
