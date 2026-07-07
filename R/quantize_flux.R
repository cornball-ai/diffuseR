#' FLUX Transformer Quantization and Loading
#'
#' Quantize the 12B FLUX transformer to NF4 (~7 GB, GPU-resident on
#' 16 GB cards) or fp8 (~12 GB, CPU-resident and streamed per forward),
#' and load any format back into \code{\link{flux_transformer}}. Reuses
#' the LTX-2.3 quantization machinery (\code{ltx23_nf4_quantize},
#' \code{ltx23_nf4_linear}, \code{ltx23_fp8_linear}); only the cast set
#' and the diffusers directory layout are FLUX-specific.
#'
#' @name quantize_flux
NULL

.flux_dtype <- function(dtype) {
    switch(dtype, bfloat16 = torch::torch_bfloat16(),
           float16 = torch::torch_float16(), float32 = torch::torch_float32(),
           stop("Unsupported dtype: ", dtype))
}

# Transformer constructor arguments from a diffusers config.json
.flux_transformer_args <- function(config) {
    if (is.null(config)) {
        return(list())
    }
    if (isTRUE(config$guidance_embeds)) {
        stop("This checkpoint uses guidance embeddings (FLUX.1-dev); ",
             "only FLUX.1-schnell (guidance_embeds = false) is supported.")
    }
    args <- list(
                 in_channels = config$in_channels,
                 num_layers = config$num_layers,
                 num_single_layers = config$num_single_layers,
                 attention_head_dim = config$attention_head_dim,
                 num_attention_heads = config$num_attention_heads,
                 joint_attention_dim = config$joint_attention_dim,
                 pooled_projection_dim = config$pooled_projection_dim,
                 axes_dims_rope = config$axes_dims_rope,
                 out_channels = config$out_channels
    )
    # JSON roundtrips turn null into empty lists; drop both
    args <- Filter(function(x) !is.null(x) && length(x) > 0L, args)
    lapply(args, function(x) if (is.numeric(x)) as.integer(x) else x)
}

.flux2_transformer_args <- function(config) {
    if (is.null(config)) {
        return(list())
    }
    if (isTRUE(config$guidance_embeds)) {
        stop("This checkpoint uses guidance embeddings (FLUX.2-dev); ",
             "only the klein variants (guidance_embeds = false) are supported.")
    }
    args <- list(
                 in_channels = config$in_channels,
                 num_layers = config$num_layers,
                 num_single_layers = config$num_single_layers,
                 attention_head_dim = config$attention_head_dim,
                 num_attention_heads = config$num_attention_heads,
                 joint_attention_dim = config$joint_attention_dim,
                 timestep_guidance_channels = config$timestep_guidance_channels,
                 axes_dims_rope = config$axes_dims_rope,
                 out_channels = config$out_channels
    )
    args <- Filter(function(x) !is.null(x) && length(x) > 0L, args)
    args <- lapply(args, function(x) if (is.numeric(x)) as.integer(x) else x)
    for (field in c("mlp_ratio", "rope_theta", "eps")) {
        v <- config[[field]]
        if (!is.null(v) && length(v) == 1L) {
            args[[field]] <- as.numeric(v)
        }
    }
    args
}

# Move plain-field fp8 weights (and their scales) onto a device; used
# for resident fp8 where the whole quantized model fits on the GPU
.flux_fp8_to_device <- function(module, device) {
    for (name in names(module$children)) {
        child <- module$children[[name]]
        if (!is.null(child$weight_fp8)) {
            child$weight_fp8 <- child$weight_fp8$to(device = device)
            child$weight_scale <- child$weight_scale$to(device = device)
        }
        .flux_fp8_to_device(child, device)
    }
    invisible(module)
}

# Family-specific hooks for quantization and loading
.flux_family_hooks <- function(config) {
    if (.flux_family(config) == "flux2") {
        list(model_fn = flux2_transformer, args_fn = .flux2_transformer_args,
             is_quant_key = flux2_is_quant_key, shard_prefix = "flux2-klein")
    } else {
        list(model_fn = flux_transformer,
             args_fn = .flux_transformer_args,
             is_quant_key = flux_is_quant_key,
             shard_prefix = "flux1")
    }
}

#' Quantize a FLUX transformer to NF4 or fp8 shards
#'
#' Streams the bf16 diffusers checkpoint tensor by tensor. Cast-set
#' weights (see \code{\link{flux_is_quant_key}}) are stored as NF4
#' (packed uint8 + \code{<key>_absmax} float32 blocks) or as
#' float8_e4m3fn with an absmax/448 per-tensor \code{<key>_scale};
#' everything else is copied through unchanged. The manifest embeds the
#' transformer config, so the source checkpoint is not needed again
#' after quantization.
#'
#' @param transformer_dir Source diffusers transformer directory.
#' @param output_dir Output directory for shards + manifest (default:
#'   the per-format location under \code{tools::R_user_dir}).
#' @param format "nf4" or "fp8".
#' @param shard_bytes Numeric. Approximate shard size.
#' @param force Logical. Re-quantize even if a valid manifest exists.
#' @param verbose Logical.
#'
#' @return Invisibly, the manifest list.
#'
#' @export
flux_quantize <- function(transformer_dir, output_dir = NULL,
                          format = c("nf4", "fp8"), shard_bytes = 4e9,
                          force = FALSE, verbose = TRUE) {
    format <- match.arg(format)
    if (is.null(output_dir)) {
        output_dir <- file.path(tools::R_user_dir("diffuseR", "data"),
                                paste0("flux1-schnell-", format))
    }

    manifest_path <- file.path(output_dir, "manifest.json")
    if (!force && file.exists(manifest_path)) {
        manifest <- jsonlite::fromJSON(manifest_path)
        if (identical(manifest$format, format) &&
            all(file.exists(file.path(output_dir, manifest$shards)))) {
            if (verbose) {
                message(toupper(format), " artifact already present: ",
                        output_dir)
            }
            return(invisible(manifest))
        }
    }
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

    ckpt <- flux_open_checkpoint(transformer_dir)
    hooks <- .flux_family_hooks(ckpt$config)
    if (format == "fp8") {
        fp8 <- torch::torch_float8_e4m3fn()
    }

    shard <- list()
    shard_size <- 0
    shard_files <- character(0)
    n_cast <- 0L

    flush_shard <- function() {
        if (!length(shard)) {
            return()
        }
        fname <- sprintf("%s-%s-%05d.safetensors", hooks$shard_prefix,
                         format, length(shard_files) + 1L)
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

    keys <- ckpt$keys
    for (i in seq_along(keys)) {
        key <- keys[[i]]
        tensor <- ckpt$handle$get_tensor(key)

        if (hooks$is_quant_key(key)) {
            torch::with_no_grad({
                if (format == "nf4") {
                    q <- ltx23_nf4_quantize(tensor)
                    shard[[key]] <- q$packed
                    shard[[paste0(key, "_absmax")]] <- q$absmax
                    shard_size <- shard_size + prod(tensor$shape) * 0.5625
                } else {
                    scale <- tensor$abs()$max()$to(dtype = torch::torch_float32())$
                    clamp(min = 1e-12) / 448
                    shard[[key]] <- (tensor$to(dtype = torch::torch_float32()) / scale)$to(dtype = fp8)
                    shard[[paste0(key, "_scale")]] <- scale
                    shard_size <- shard_size + prod(tensor$shape)
                }
            })
            n_cast <- n_cast + 1L
        } else {
            shard[[key]] <- tensor
            shard_size <- shard_size + prod(tensor$shape) * 2
        }
        rm(tensor)

        if (shard_size >= shard_bytes) {
            flush_shard()
        }
        if (i %% 200L == 0L) {
            gc(verbose = FALSE)
            if (verbose) {
                message(sprintf("  quantizing %d/%d tensors", i, length(keys)))
            }
        }
    }
    flush_shard()

    manifest <- list(
                     source = basename(transformer_dir),
                     format = format,
                     shards = shard_files,
                     tensors = length(keys),
                     cast = n_cast,
                     config = ckpt$config
    )
    jsonlite::write_json(manifest, manifest_path, auto_unbox = TRUE,
                         pretty = TRUE)
    if (verbose) {
        message(sprintf("Quantized %d/%d tensors to %s across %d shards: %s",
                        n_cast, length(keys), format, length(shard_files),
                        output_dir))
    }
    invisible(manifest)
}

#' Load a FLUX transformer from any checkpoint format
#'
#' Builds \code{\link{flux_transformer}} from the checkpoint's embedded
#' config and loads the weights. Dispatches on the checkpoint format:
#'
#' \itemize{
#'   \item full precision (\code{\link{flux_open_checkpoint}}): weights
#'     stream into the model in \code{dtype} on \code{device}.
#'   \item \code{"nf4"} (\code{\link{flux_open_quantized}}): cast-set
#'     linears become \code{ltx23_nf4_linear}; the whole model (packed
#'     weights included) moves to \code{device} and stays resident.
#'   \item \code{"fp8"}: cast-set linears become
#'     \code{ltx23_fp8_linear}; fp8 weights stay CPU-resident (optionally
#'     pinned) and stream to \code{device} inside each forward.
#' }
#'
#' @param ckpt A checkpoint from \code{\link{flux_open_checkpoint}} or
#'   \code{\link{flux_open_quantized}}.
#' @param device Character. Compute device.
#' @param dtype Character. Model dtype ("bfloat16" or "float32"). For
#'   quantized formats this sets the resident (non-quantized) tensors
#'   and must match the compute dtype: bfloat16 for GPU compute,
#'   float32 for CPU compute.
#' @param pin Logical. Pin fp8 host memory for faster transfers
#'   (streamed fp8 only).
#' @param fp8_resident Logical. Keep the fp8 weights on \code{device}
#'   instead of streaming from the CPU - right for models whose whole
#'   quantized footprint fits in VRAM (FLUX.2 klein-4B: ~4 GB).
#' @param verbose Logical.
#' @param ... Overrides for \code{\link{flux_transformer}} arguments
#'   (tiny test configs).
#'
#' @return The loaded \code{flux_transformer} in eval mode.
#'
#' @export
flux_load_transformer <- function(ckpt, device = "cuda", dtype = "bfloat16",
                                  pin = TRUE, fp8_resident = FALSE,
                                  verbose = TRUE, ...) {
    stopifnot(inherits(ckpt, "ltx23_checkpoint"))
    format <- ckpt$format %||% "full"
    hooks <- .flux_family_hooks(ckpt$config)

    args <- utils::modifyList(hooks$args_fn(ckpt$config), list(...))
    model <- do.call(hooks$model_fn, args)

    if (format == "full") {
        model$to(dtype = .flux_dtype(dtype))
        res <- ltx23_load_group(ckpt, ckpt$keys, model, verbose = verbose)
        if (length(res$unmapped) || length(res$unfilled)) {
            stop("FLUX transformer load: ", length(res$unmapped),
                 " unmapped keys, ", length(res$unfilled), " unfilled params")
        }
        model$to(device = device)
        model$eval()
        return(model)
    }

    if (!format %in% c("nf4", "fp8")) {
        stop("Unknown checkpoint format: ", format)
    }
    # Residents (embedders, norms, biases) in the compute dtype: the
    # quantized linears dequantize into the input's dtype at forward, so
    # the two must agree (bfloat16 on GPU, float32 for CPU compute)
    model$to(dtype = .flux_dtype(dtype))

    if (format == "nf4") {
        sib_suffix <- "_absmax"
    } else {
        sib_suffix <- "_scale"
    }
    sib_keys <- ckpt$keys[endsWith(ckpt$keys, paste0(".weight", sib_suffix))]
    main_keys <- setdiff(ckpt$keys, sib_keys)

    dests <- c(model$named_parameters(), model$named_buffers())
    filled <- character(0)
    unmapped <- character(0)

    torch::with_no_grad({
        for (i in seq_along(main_keys)) {
            key <- main_keys[[i]]

            if (hooks$is_quant_key(key) &&
                        paste0(key, sib_suffix) %in% sib_keys) {
                segments <- strsplit(key, ".", fixed = TRUE)[[1]]
                parent <- .ltx23_walk_module(model, utils::head(segments, -2L))
                leaf <- segments[length(segments) - 1L]
                old <- .ltx23_walk_module(parent, leaf)
                if (is.null(old)) {
                    unmapped <- c(unmapped, key)
                    next
                }
                w_shape <- old$weight$shape
                has_bias <- !is.null(old$bias)
                quant_mod <- if (format == "nf4") {
                    ltx23_nf4_linear(w_shape[1], w_shape[2], bias = has_bias)
                } else {
                    ltx23_fp8_linear(w_shape[1], w_shape[2], bias = has_bias)
                }
                if (has_bias) {
                    # Adopt the original bias parameter; its checkpoint key
                    # loads through the pre-swap destination map
                    quant_mod$bias <- old$bias
                }
                if (format == "nf4") {
                    quant_mod$set_nf4_weight(
                        ckpt$handle$get_tensor(key),
                        ckpt$handle$get_tensor(paste0(key, sib_suffix))
                    )
                } else {
                    quant_mod$set_fp8_weight(
                        ckpt$handle$get_tensor(key),
                        ckpt$handle$get_tensor(paste0(key, sib_suffix)),
                        pin = pin
                    )
                }
                do.call(`$<-`, list(parent, leaf, quant_mod))
                filled <- c(filled, key)
            } else {
                dest <- dests[[key]]
                if (is.null(dest)) {
                    unmapped <- c(unmapped, key)
                    next
                }
                dest$copy_(ckpt$handle$get_tensor(key))
                filled <- c(filled, key)
            }

            if (i %% 100L == 0L) {
                gc(verbose = FALSE)
                if (verbose && i %% 500L == 0L) {
                    message(sprintf("  loaded %d/%d transformer tensors", i,
                                    length(main_keys)))
                }
            }
        }
    })
    gc(verbose = FALSE)

    if (length(unmapped)) {
        stop("FLUX ", format, " load: ", length(unmapped),
             " unmapped keys, e.g. ",
             paste(utils::head(unmapped, 3), collapse = ", "))
    }
    # Weight params replaced by quantized modules won't be "filled"
    expected_missing <- hooks$is_quant_key(names(dests))
    unfilled <- setdiff(names(dests)[!expected_missing], filled)
    if (length(unfilled)) {
        stop("FLUX ", format, " load: ", length(unfilled),
             " unfilled params, e.g. ",
             paste(utils::head(unfilled, 3), collapse = ", "))
    }

    # NF4: everything (packed buffers included) onto the device.
    # FP8: residents move; the plain-field fp8 weights stay CPU-side
    # unless fp8_resident moves them (small models that fit on the GPU).
    model$to(device = device)
    if (format == "fp8" && fp8_resident) {
        .flux_fp8_to_device(model, device)
    }
    model$eval()
    # Block intermediates are large at image resolutions; per-block gc
    # keeps the quantized-linear temporaries bounded
    options(diffuseR.block_gc = TRUE)
    if (verbose) {
        message("FLUX transformer ready (", format, ") on ", device)
    }
    model
}
