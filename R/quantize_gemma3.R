#' Quantize a Gemma3 text encoder to NF4 shards
#'
#' Streams the HuggingFace Gemma3 checkpoint tensor by tensor. The
#' language model's projection weights (q/k/v/o and gate/up/down, ~11B
#' of the 12B parameters) are stored as NF4 (packed uint8 +
#' \code{<key>_absmax} float32 blocks); embeddings and norms are copied
#' at the resident dtype. Vision-tower and projector weights are
#' dropped - the text encoder never uses them. The result is a ~8 GB
#' artifact that fits a 16 GB card during the encode phase (vs 45 GB
#' of host RAM for the fp32 CPU path).
#'
#' Keys are stored normalized (\code{language_model.} / \code{model.}
#' prefixes stripped), matching the module tree of
#' \code{\link{gemma3_text_model}}.
#'
#' @param model_path HuggingFace snapshot directory (config.json +
#'   model-*.safetensors).
#' @param output_dir Output directory for shards + manifest (default:
#'   \code{gemma3-nf4} under \code{tools::R_user_dir}).
#' @param shard_bytes Numeric. Target shard size in bytes; the 1.9e9
#'   default keeps shards readable by stock CRAN safetensors.
#' @param force Logical. Re-quantize even if a valid manifest exists.
#' @param verbose Logical.
#'
#' @return Invisibly, the manifest list.
#'
#' @export
gemma3_quantize_nf4 <- function(model_path, output_dir = NULL,
                                shard_bytes = 1.9e9, force = FALSE,
                                verbose = TRUE) {
    model_path <- path.expand(model_path)
    if (is.null(output_dir)) {
        output_dir <- file.path(tools::R_user_dir("diffuseR", "data"),
                                "gemma3-nf4")
    }

    manifest_path <- file.path(output_dir, "manifest.json")
    if (!force && file.exists(manifest_path)) {
        manifest <- jsonlite::fromJSON(manifest_path)
        if (identical(manifest$format, "nf4") &&
            all(file.exists(file.path(output_dir, manifest$shards)))) {
            if (verbose) {
                message("NF4 artifact already present: ", output_dir)
            }
            return(invisible(manifest))
        }
    }

    config_path <- file.path(model_path, "config.json")
    if (!file.exists(config_path)) {
        stop("Config file not found: ", config_path)
    }
    config_raw <- jsonlite::fromJSON(config_path)
    if (!is.null(config_raw$text_config)) {
        config_raw <- config_raw$text_config
    }

    resident_dtype <- if (.st_can_write("bfloat16")) {
        torch::torch_bfloat16()
    } else {
        message("safetensors cannot write bfloat16; storing resident ",
                "tensors as float32 (larger artifact, same results)")
        torch::torch_float32()
    }

    opened <- .flux_open_sharded_dir(model_path, "model")
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

    shard <- list()
    shard_size <- 0
    shard_files <- character(0)
    n_cast <- 0L

    flush_shard <- function() {
        if (!length(shard)) {
            return()
        }
        fname <- sprintf("gemma3-nf4-%05d.safetensors",
                         length(shard_files) + 1L)
        safetensors::safe_save_file(shard, file.path(output_dir, fname))
        shard_files[[length(shard_files) + 1L]] <<- fname
        if (verbose) {
            message(sprintf("  wrote %s (%.2f GB, %d tensors)", fname,
                            shard_size / 1e9, length(shard)))
        }
        shard <<- list()
        shard_size <<- 0
        gc(verbose = FALSE)
    }

    for (key in opened$keys) {
        if (grepl("^(vision_tower|multi_modal_projector|lm_head)\\.", key)) {
            next
        }
        norm_key <- sub("^model\\.", "", sub("^language_model\\.", "", key))
        tensor <- opened$handle$get_tensor(key)

        if (.gemma3_is_quant_key(norm_key)) {
            torch::with_no_grad({
                q <- ltx23_nf4_quantize(tensor)
                shard[[norm_key]] <- q$packed
                shard[[paste0(norm_key, "_absmax")]] <- q$absmax
            })
            shard_size <- shard_size + prod(tensor$shape) * 0.5625
            n_cast <- n_cast + 1L
        } else {
            shard[[norm_key]] <- tensor$to(dtype = resident_dtype)
            shard_size <- shard_size + prod(tensor$shape) * 2
        }
        rm(tensor)
        if (shard_size >= shard_bytes) {
            flush_shard()
        }
    }
    flush_shard()

    manifest <- list(model = "gemma3", format = "nf4", shards = shard_files,
                     n_cast = n_cast, config = config_raw)
    jsonlite::write_json(manifest, manifest_path, auto_unbox = TRUE,
                         pretty = TRUE)
    if (verbose) {
        message(sprintf("Quantized %d weights to NF4 across %d shards: %s",
                        n_cast, length(shard_files), output_dir))
    }
    invisible(manifest)
}

# The 7 per-layer projection weights carrying ~11B of the 12B params
.gemma3_is_quant_key <- function(key) {
    grepl(paste0("^layers\\.[0-9]+\\.(self_attn\\.(q|k|v|o)_proj|",
                 "mlp\\.(gate|up|down)_proj)\\.weight$"), key)
}

#' Load a Gemma3 text encoder from an NF4 artifact
#'
#' Builds the model as a skeleton at the compute dtype, swaps the
#' projection linears for NF4 modules filled from the artifact
#' (dequantized per forward through the shared byte-LUT), copies the
#' residents, and hard-errors on any parameter the artifact does not
#' fill.
#'
#' @param artifact_dir Directory produced by
#'   \code{\link{gemma3_quantize_nf4}}.
#' @param device "cuda" or "cpu".
#' @param dtype Compute dtype ("bfloat16" default).
#' @param pin Logical. When loading to the CPU, page-lock the weights
#'   so \code{\link{encode_with_gemma3}} can swap the model to the GPU
#'   at DMA speed per encode and back for free (see
#'   \code{\link{staging}}). Default follows
#'   \code{options(diffuseR.pin_staging)}.
#' @param verbose Logical.
#'
#' @return A \code{gemma3_text_model} ready for
#'   \code{\link{encode_with_gemma3}}.
#'
#' @export
load_gemma3_nf4 <- function(artifact_dir, device = "cuda",
                            dtype = "bfloat16",
                            pin = getOption("diffuseR.pin_staging", TRUE),
                            verbose = TRUE) {
    ckpt <- ltx23_open_fp8_checkpoint(artifact_dir)
    manifest <- jsonlite::fromJSON(file.path(artifact_dir, "manifest.json"))
    if (!identical(manifest$model, "gemma3")) {
        stop("Not a Gemma3 NF4 artifact (manifest model is ",
             manifest$model %||% "absent", "): ", artifact_dir)
    }
    config <- .gemma3_build_config(manifest$config)

    torch_dtype <- switch(dtype,
                          "float32" = torch::torch_float32(),
                          "float16" = torch::torch_float16(),
                          "bfloat16" = torch::torch_bfloat16(),
                          torch::torch_bfloat16())

    if (verbose) {
        message(sprintf("Creating Gemma3 NF4 model: %d layers, hidden_size=%d",
                        config$num_hidden_layers, config$hidden_size))
    }
    model <- .construct_skeleton(gemma3_text_model, config, dtype = torch_dtype)

    cast_keys <- grep("_absmax$", ckpt$keys, value = TRUE)
    cast_keys <- sub("_absmax$", "", cast_keys)
    torch::with_no_grad({
        for (key in cast_keys) {
            segs <- strsplit(sub("\\.weight$", "", key), ".", fixed = TRUE)[[1]]
            # layers.<l>.<group>.<proj>
            layer <- model$layers[[as.integer(segs[2]) + 1L]]
            parent <- layer[[segs[3]]]
            old <- parent[[segs[4]]]
            nf4 <- ltx23_nf4_linear(old$weight$shape[1], old$weight$shape[2],
                                    bias = FALSE)
            nf4$set_nf4_weight(ckpt$handle$get_tensor(key),
                               ckpt$handle$get_tensor(paste0(key, "_absmax")))
            do.call(`$<-`, list(parent, segs[4], nf4))
        }

        dests <- model$parameters
        filled <- character(0)
        for (key in setdiff(ckpt$keys, c(cast_keys,
                    paste0(cast_keys, "_absmax")))) {
            dest <- dests[[key]]
            if (is.null(dest)) {
                next
            }
            dest$copy_(ckpt$handle$get_tensor(key))
            filled <- c(filled, key)
        }
        unfilled <- setdiff(names(dests), filled)
        if (length(unfilled)) {
            stop("Gemma3 NF4 load: ", length(unfilled),
                 " parameters not filled, e.g. ",
                 paste(utils::head(unfilled, 3), collapse = ", "))
        }
    })

    model$to(device = device)
    model$eval()
    if (pin && device == "cpu") {
        st <- .pin_component(model)
        if (!is.null(st)) {
            attr(model, "staging") <- st
            if (verbose) {
                message("Gemma3 weights pinned for staged transfer")
            }
        }
    }
    if (verbose) {
        message("Gemma3 NF4 encoder ready on ", device)
    }
    model
}
