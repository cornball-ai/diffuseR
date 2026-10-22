#' Quantize the LTX-2.5 Gemma 4 text encoder to NF4 shards
#'
#' Streams the LTX-2.5 encoder file tensor by tensor. The per-layer
#' projection weights (q/k/v/o and gate/up/down; 328 tensors in the 12B,
#' since full-attention layers have no v_proj) are stored as NF4 (packed
#' uint8 + \code{<key>_absmax} float32 blocks); embeddings, norms and the
#' per-layer \code{layer_scalar} are copied at the resident dtype. Vision
#' and audio towers are dropped.
#'
#' The artifact is self-contained for the LTX-2.5 text side: it also
#' keeps the connectors' \code{text_embedding_projection.*} weights
#' (resident dtype, original keys) and writes the embedded tokenizer as
#' \code{tokenizer.json}, so the 26 GB source file is not needed after
#' quantizing. Load it with \code{\link{load_gemma4_nf4}} (or
#' \code{\link{load_gemma4_text_encoder}} on the directory) and the
#' tokenizer with \code{gemma4_tokenizer(output_dir)}.
#'
#' Peak host memory is about one shard (\code{shard_bytes}) plus the
#' largest single source tensor (the 2 GB bf16 embedding).
#'
#' @param path The LTX-2.5 text encoder file
#'   (\code{gemma4-12b-with-proj-ltx-2.5-bf16.safetensors}).
#' @param output_dir Output directory (default: \code{gemma4-nf4} under
#'   \code{tools::R_user_dir}).
#' @param shard_bytes Numeric. Target shard size in bytes; the 1.9e9
#'   default keeps shards readable by safetensors older than 0.3.0.
#' @param force Logical. Re-quantize even if a valid manifest exists.
#' @param verbose Logical.
#'
#' @return Invisibly, the manifest list.
#'
#' @export
gemma4_quantize_nf4 <- function(path, output_dir = NULL, shard_bytes = 1.9e9,
                                force = FALSE, verbose = TRUE) {
    if (is.null(output_dir)) {
        output_dir <- file.path(tools::R_user_dir("diffuseR", "data"),
                                "gemma4-nf4")
    }
    manifest_path <- file.path(output_dir, "manifest.json")
    if (!force && file.exists(manifest_path)) {
        manifest <- jsonlite::fromJSON(manifest_path)
        if (identical(manifest$model, "gemma4") &&
            identical(manifest$format, "nf4") &&
            all(file.exists(file.path(output_dir, manifest$shards)))) {
            if (verbose) {
                message("NF4 artifact already present: ", output_dir)
            }
            return(invisible(manifest))
        }
    }

    ckpt <- ltx23_open_checkpoint(path, require_version = NULL)
    meta <- ckpt$handle$metadata[["__metadata__"]]
    if (is.null(meta[["gemma_config"]])) {
        stop("No gemma_config metadata in ", path,
             "; expected an LTX-2.5 Gemma 4 text encoder file")
    }
    config_raw <- jsonlite::fromJSON(meta[["gemma_config"]],
                                     simplifyVector = FALSE)
    resident_dtype <- if (.st_can_write("bfloat16")) {
        torch::torch_bfloat16()
    } else {
        message("safetensors cannot write bfloat16; storing resident ",
                "tensors as float32 (larger artifact, same results)")
        torch::torch_float32()
    }
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    # A stale manifest must not validate a half-written artifact
    unlink(manifest_path)

    if ("tokenizer_json" %in% ckpt$keys) {
        writeLines(.gemma4_tokenizer_json(path),
                   file.path(output_dir, "tokenizer.json"), useBytes = TRUE)
    }

    shard <- list()
    shard_size <- 0
    shard_files <- character(0)
    n_cast <- 0L
    flush_shard <- function() {
        if (!length(shard)) {
            return()
        }
        fname <- sprintf("gemma4-nf4-%05d.safetensors", length(shard_files) + 1L)
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

    for (key in ckpt$keys) {
        name <- .gemma4_map_key(key)
        if (is.na(name)) {
            if (!startsWith(key, "text_embedding_projection.")) {
                next
            }
            name <- key # kept for the LTX-2.5 connectors
        }
        tensor <- ckpt$handle$get_tensor(key)
        if (.gemma3_is_quant_key(name)) {
            torch::with_no_grad({
                q <- ltx23_nf4_quantize(tensor)
                shard[[name]] <- q$packed
                shard[[paste0(name, "_absmax")]] <- q$absmax
            })
            shard_size <- shard_size + prod(tensor$shape) * 0.5625
            n_cast <- n_cast + 1L
        } else {
            shard[[name]] <- tensor$to(dtype = resident_dtype)
            shard_size <- shard_size + prod(tensor$shape) * 2
        }
        rm(tensor)
        if (shard_size >= shard_bytes) {
            flush_shard()
        }
    }
    flush_shard()

    manifest <- list(model = "gemma4", format = "nf4", shards = shard_files,
                     n_cast = n_cast, config = config_raw)
    jsonlite::write_json(manifest, manifest_path, auto_unbox = TRUE,
                         pretty = TRUE, digits = NA, null = "null")
    if (verbose) {
        message(sprintf("Quantized %d weights to NF4 across %d shards: %s",
                        n_cast, length(shard_files), output_dir))
    }
    invisible(manifest)
}

#' Load the LTX-2.5 Gemma 4 text encoder from an NF4 artifact
#'
#' Builds the model as a skeleton at the compute dtype, swaps the
#' projection linears for NF4 modules filled from the artifact
#' (dequantized per forward), copies the residents (parameters and the
#' \code{layer_scalar} buffers), and hard-errors on anything the artifact
#' does not fill. The connectors' projection weights in the artifact are
#' left for the connector loader.
#'
#' @param artifact_dir Directory produced by \code{\link{gemma4_quantize_nf4}}.
#' @param device "cuda" or "cpu".
#' @param dtype Compute dtype ("bfloat16" default, or "float32").
#' @param pin Logical. When loading to the CPU, page-lock the weights so
#'   \code{\link{encode_with_gemma4}} can move the model to the GPU per
#'   encode at DMA speed. Default follows
#'   \code{options(diffuseR.pin_staging)}.
#' @param verbose Logical.
#'
#' @return A \code{gemma4_text_model}.
#'
#' @export
load_gemma4_nf4 <- function(artifact_dir, device = "cuda", dtype = "bfloat16",
                            pin = getOption("diffuseR.pin_staging", TRUE),
                            verbose = TRUE) {
    ckpt <- ltx23_open_fp8_checkpoint(artifact_dir)
    manifest <- jsonlite::fromJSON(file.path(artifact_dir, "manifest.json"),
                                   simplifyVector = FALSE)
    if (!identical(manifest$model, "gemma4")) {
        stop("Not a Gemma 4 NF4 artifact (manifest model is ",
             manifest$model %||% "absent", "): ", artifact_dir)
    }
    config <- .gemma4_build_config(manifest$config)
    torch_dtype <- switch(dtype, bfloat16 = torch::torch_bfloat16(),
                          float32 = torch::torch_float32(),
                          stop("Unsupported dtype: ", dtype))
    if (verbose) {
        message(sprintf("Creating Gemma 4 NF4 model: %d layers, hidden_size=%d",
                        config$num_hidden_layers, config$hidden_size))
    }
    model <- .construct_skeleton(gemma4_text_model, config, dtype = torch_dtype)

    cast_keys <- sub("_absmax$", "", grep("_absmax$", ckpt$keys, value = TRUE))
    torch::with_no_grad({
        for (key in cast_keys) {
            # layers.<l>.<self_attn|mlp>.<proj>.weight
            segs <- strsplit(sub("\\.weight$", "", key), ".", fixed = TRUE)[[1]]
            parent <- model$layers[[as.integer(segs[2]) + 1L]][[segs[3]]]
            old <- parent[[segs[4]]]
            nf4 <- ltx23_nf4_linear(old$weight$shape[1], old$weight$shape[2],
                                    bias = FALSE)
            nf4$set_nf4_weight(ckpt$handle$get_tensor(key),
                               ckpt$handle$get_tensor(paste0(key, "_absmax")))
            do.call(`$<-`, list(parent, segs[4], nf4))
        }

        dests <- c(model$named_parameters(), model$named_buffers())
        dests <- dests[!grepl("weight_(nf4|absmax)$", names(dests))]
        filled <- character(0)
        residents <- setdiff(ckpt$keys, c(cast_keys, paste0(cast_keys, "_absmax")))
        for (key in residents) {
            dest <- dests[[key]]
            if (is.null(dest)) {
                next # e.g. text_embedding_projection.*, for the connectors
            }
            dest$copy_(ckpt$handle$get_tensor(key))
            filled <- c(filled, key)
        }
        unfilled <- setdiff(names(dests), filled)
        if (length(unfilled)) {
            stop("Gemma 4 NF4 load: ", length(unfilled),
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
                message("Gemma 4 weights pinned for staged transfer")
            }
        }
    }
    model
}
