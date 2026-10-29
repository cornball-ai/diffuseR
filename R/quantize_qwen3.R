#' Quantize a Qwen3 text encoder to fp8 or NF4 shards
#'
#' Streams a transformers Qwen3 (FLUX.2 klein, Z-Image) or Qwen3-VL
#' (FLUX 3) text encoder tensor by tensor. The per-layer projection
#' weights (q/k/v/o and gate/up/down, 252 tensors in the 36-layer 4B)
#' are stored as float8_e4m3fn with a per-tensor absmax/448
#' \code{<key>_scale}, or as NF4 (packed uint8 + \code{<key>_absmax});
#' embeddings and norms are copied at bfloat16. The vision tower and the
#' tied LM head are dropped, and keys are written in the
#' \code{\link{qwen3_encoder}} layout (\code{model.layers.*}). The
#' manifest embeds the text config, so the loader needs no other file.
#'
#' Host memory: the 4B text model is ~8 GB in bfloat16, ~4.4 GB as fp8
#' and ~2.8 GB as NF4 (the 0.8 GB embedding stays bfloat16 in both).
#' Load the artifact with \code{\link{load_qwen3_text_encoder}} on its
#' directory.
#'
#' @param model_path Source directory with \code{config.json} and
#'   \code{model*.safetensors}.
#' @param output_dir Output directory for shards + manifest.
#' @param format "fp8" or "nf4".
#' @param shard_bytes Numeric. Target shard size in bytes (1.9e9 keeps
#'   shards readable by safetensors older than 0.3.0).
#' @param force Logical. Re-quantize even if a valid manifest exists.
#' @param verbose Logical.
#'
#' @return Invisibly, the manifest list.
#'
#' @export
qwen3_quantize <- function(model_path, output_dir, format = c("fp8", "nf4"),
                           shard_bytes = 1.9e9, force = FALSE,
                           verbose = TRUE) {
    format <- match.arg(format)
    if (format == "fp8" && !.st_can_write("float8_e4m3fn")) {
        stop("The installed safetensors package cannot write float8 ",
             "tensors (released in safetensors 0.3.0). Run ",
             "install.packages(\"safetensors\") to update, or use ",
             "format = \"nf4\".", call. = FALSE)
    }
    manifest_path <- file.path(output_dir, "manifest.json")
    if (!force && .is_qwen3_artifact(output_dir, format)) {
        if (verbose) {
            message(toupper(format), " Qwen3 artifact already present: ",
                    output_dir)
        }
        return(invisible(jsonlite::fromJSON(manifest_path,
                                            simplifyVector = FALSE)))
    }

    model_path <- path.expand(model_path)
    config <- jsonlite::fromJSON(file.path(model_path, "config.json"),
                                 simplifyVector = FALSE)
    vl <- !is.null(config$text_config)
    text_config <- if (vl) config$text_config else config
    resident_dtype <- if (.st_can_write("bfloat16")) {
        torch::torch_bfloat16()
    } else {
        message("safetensors cannot write bfloat16; storing resident ",
                "tensors as float32 (larger artifact, same results)")
        torch::torch_float32()
    }
    resident_bytes <- torch::torch_empty(1L, dtype = resident_dtype)$element_size()
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    # A stale manifest must not validate a half-written artifact
    unlink(manifest_path)

    opened <- .flux_open_sharded_dir(model_path, "model")
    shard <- list()
    shard_size <- 0
    shard_files <- character(0)
    n_cast <- 0L
    flush_shard <- function() {
        if (!length(shard)) {
            return()
        }
        fname <- sprintf("qwen3-%s-%05d.safetensors", format,
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
        name <- .qwen3_map_key(key, vl)
        if (is.na(name)) {
            next
        }
        tensor <- opened$handle$get_tensor(key)
        quant <- .qwen3_is_quant_key(name)
        per_elem <- if (!quant) {
            resident_bytes
        } else if (format == "fp8") {
            1
        } else {
            0.5625
        }
        bytes <- prod(tensor$shape) * per_elem
        if (shard_size + bytes > shard_bytes) {
            flush_shard()
        }
        torch::with_no_grad({
            if (quant && format == "fp8") {
                q <- .fp8_quantize(tensor)
                shard[[name]] <- q$weight
                shard[[paste0(name, "_scale")]] <- q$scale
            } else if (quant) {
                q <- ltx23_nf4_quantize(tensor)
                shard[[name]] <- q$packed
                shard[[paste0(name, "_absmax")]] <- q$absmax
            } else {
                shard[[name]] <- tensor$to(dtype = resident_dtype)
            }
        })
        n_cast <- n_cast + quant
        shard_size <- shard_size + bytes
        rm(tensor)
        # torch's native memory is invisible to R's gc trigger: collect
        # per tensor so the source tensors and casts do not pile up
        gc(verbose = FALSE)
    }
    flush_shard()

    manifest <- list(model = "qwen3", format = format, shards = shard_files,
                     cast = n_cast, config = text_config)
    jsonlite::write_json(manifest, manifest_path, auto_unbox = TRUE,
                         pretty = TRUE, digits = NA, null = "null")
    if (verbose) {
        message(sprintf("Quantized %d Qwen3 weights to %s across %d shards: %s",
                        n_cast, format, length(shard_files), output_dir))
    }
    invisible(manifest)
}

# transformers key -> qwen3_encoder name; NA for keys the text model does
# not use (tied LM head, Qwen3-VL vision tower)
.qwen3_map_key <- function(key, vl) {
    if (identical(key, "lm_head.weight")) {
        return(NA_character_)
    }
    if (vl) {
        if (startsWith(key, "model.visual.")) {
            return(NA_character_)
        }
        key <- sub("^model\\.language_model\\.", "model.", key)
    }
    key
}

# The quantized set: every decoder layer's attention and MLP projections
.qwen3_is_quant_key <- function(name) {
    grepl(paste0("^model\\.layers\\.[0-9]+\\.",
                 "(self_attn\\.[qkvo]_proj|mlp\\.(gate|up|down)_proj)\\.weight$"),
          name)
}

# TRUE when `dir` holds a complete Qwen3 artifact (of `format`, if given)
.is_qwen3_artifact <- function(dir, format = NULL) {
    manifest_path <- file.path(dir, "manifest.json")
    if (!file.exists(manifest_path)) {
        return(FALSE)
    }
    manifest <- jsonlite::fromJSON(manifest_path)
    identical(manifest$model, "qwen3") &&
        (is.null(format) || identical(manifest$format, format)) &&
        all(file.exists(file.path(dir, manifest$shards)))
}

# Build a qwen3_encoder from a quantized artifact: skeleton at the
# compute dtype, the cast-set linears swapped for fp8 or NF4 modules
# (dequantized per forward), residents copied, anything unfilled an
# error. fp8 weights are plain module fields: they follow the module to
# `device` here, and staging must collect them (.flux_fp8_collect).
.load_qwen3_quantized <- function(dir, device, dtype, verbose, ...) {
    manifest <- jsonlite::fromJSON(file.path(dir, "manifest.json"),
                                   simplifyVector = TRUE)
    format <- manifest$format
    if (identical(format, "fp8") && !.st_can_write("float8_e4m3fn")) {
        stop("This fp8 artifact needs float8 support the installed ",
             "safetensors lacks (released in safetensors 0.3.0).",
             call. = FALSE)
    }
    args <- utils::modifyList(.qwen3_args(manifest$config), list(...))
    if (verbose) {
        message(sprintf("Creating Qwen3 %s model: %d layers", format,
                        args$num_hidden_layers %||% 36L))
    }
    model <- do.call(.construct_skeleton,
                     c(list(qwen3_encoder), args, list(dtype = .flux_dtype(dtype))))
    ckpt <- ltx23_open_fp8_checkpoint(dir)
    sib_suffix <- if (format == "fp8") "_scale" else "_absmax"
    cast_keys <- sub(paste0(sib_suffix, "$"), "",
                     grep(paste0("\\.weight", sib_suffix, "$"), ckpt$keys,
                          value = TRUE))

    torch::with_no_grad({
        for (key in cast_keys) {
            segs <- strsplit(key, ".", fixed = TRUE)[[1]]
            parent <- .ltx23_walk_module(model, utils::head(segs, -2L))
            leaf <- segs[length(segs) - 1L]
            shape <- parent[[leaf]]$weight$shape
            w <- ckpt$handle$get_tensor(key)
            sib <- ckpt$handle$get_tensor(paste0(key, sib_suffix))
            # Assign the module itself: the setters return the inner R6
            # object, which `$<-` does not register as a child
            if (format == "fp8") {
                q <- ltx23_fp8_linear(shape[1], shape[2], bias = FALSE)
                q$set_fp8_weight(w, sib)
            } else {
                q <- ltx23_nf4_linear(shape[1], shape[2], bias = FALSE)
                q$set_nf4_weight(w, sib)
            }
            do.call(`$<-`, list(parent, leaf, q))
        }

        dests <- c(model$named_parameters(), model$named_buffers())
        dests <- dests[!grepl("weight_(nf4|absmax)$", names(dests))]
        residents <- setdiff(ckpt$keys,
                             c(cast_keys, paste0(cast_keys, sib_suffix)))
        unmapped <- setdiff(residents, names(dests))
        unfilled <- setdiff(names(dests), residents)
        if (length(unmapped) || length(unfilled)) {
            stop("Qwen3 ", format, " load: ", length(unmapped),
                 " unmapped keys, ", length(unfilled), " unfilled params")
        }
        for (key in residents) {
            dests[[key]]$copy_(ckpt$handle$get_tensor(key))
        }
    })
    gc(verbose = FALSE)

    model$to(device = device)
    if (format == "fp8" && device != "cpu") {
        .flux_fp8_to_device(model, device)
    }
    model$eval()
    model
}
