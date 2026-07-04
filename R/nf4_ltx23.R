#' NF4 Weight Storage for the LTX-2.3 Transformer
#'
#' 4-bit NormalFloat quantization (the QLoRA scheme: per-block absmax
#' normalization against a 16-level quantile code, two indices packed
#' per byte). At ~4.5 bits/parameter the 22B transformer fits in about
#' 12.5 GB, small enough to stay resident on a 16 GB GPU: no per-step
#' PCIe weight streaming, at a small quality cost relative to fp8.
#' Quantization and dequantization are pure torch ops (bucketize,
#' index_select) - no custom kernels.
#'
#' @name nf4_ltx23
NULL

# The 16 NF4 quantile levels (QLoRA, Dettmers et al. 2023)
.ltx23_nf4_table <- c(-1.0, -0.6961928009986877, -0.5250730514526367,
                      -0.39491748809814453, -0.28444138169288635,
                      -0.18477343022823334, -0.09105003625154495, 0.0,
                      0.07958029955625534, 0.16093020141124725,
                      0.24611230194568634, 0.33791524171829224,
                      0.44070982933044434, 0.5626170039176941,
                      0.7229568362236023, 1.0)

.ltx23_nf4_block_size <- 64L

#' Quantize a tensor to NF4
#'
#' @param x Float tensor (any shape; total elements must be a multiple
#'   of 128, i.e. two 64-element blocks - always true for the LTX
#'   linears).
#'
#' @return List with \code{packed} (uint8, two indices per byte) and
#'   \code{absmax} (float32, one per 64-element block).
#'
#' @export
ltx23_nf4_quantize <- function(x) {
    n <- prod(x$shape)
    block <- .ltx23_nf4_block_size
    if (n %% block != 0L) {
        stop("Tensor length must be a multiple of ", block)
    }

    table <- torch::torch_tensor(.ltx23_nf4_table,
                                 dtype = torch::torch_float32(),
                                 device = x$device)
    # Decision boundaries at the midpoints between adjacent levels
    midpoints <- (table$narrow(1L, 1L, 15L) + table$narrow(1L, 2L, 15L)) / 2

    blocks <- x$detach()$to(dtype = torch::torch_float32())$reshape(c(-1L, block))
    absmax <- blocks$abs()$amax(dim = 2L)$clamp(min = 1e-12)
    normalized <- blocks / absmax$unsqueeze(2L)

    idx <- torch::torch_bucketize(normalized$flatten(), midpoints) # 0..15
    idx <- idx$to(dtype = torch::torch_uint8())

    # Pack pairs: first index in the high nibble
    pairs <- idx$reshape(c(-1L, 2L))
    packed <- pairs$narrow(2L, 1L, 1L)$squeeze(2L) * 16L +
    pairs$narrow(2L, 2L, 1L)$squeeze(2L)

    list(packed = packed, absmax = absmax)
}

#' Dequantize NF4 data to a float tensor
#'
#' @param packed uint8 tensor of packed index pairs.
#' @param absmax float32 tensor of per-block scales.
#' @param shape Integer vector. Original tensor shape.
#' @param dtype Target torch dtype.
#' @param chunk_elements Integer. Elements dequantized per slice (bounds
#'   the int64 index temporary).
#' @param out Optional preallocated tensor of \code{shape} to write into
#'   (avoids allocating a fresh weight tensor per call).
#'
#' @return Tensor of \code{shape} in \code{dtype} on the input's device.
#'
#' @export
ltx23_nf4_dequantize <- function(packed, absmax, shape,
                                 dtype = torch::torch_bfloat16(),
                                 chunk_elements = 8388608L, out = NULL) {
    block <- .ltx23_nf4_block_size

    if (is.null(out)) {
        out <- torch::torch_empty(shape, dtype = dtype, device = packed$device)
    }
    out_flat <- out$view(-1L)

    n_bytes <- packed$shape[1]
    bytes_per_chunk <- max(chunk_elements %/% 2L, block)
    scratch <- .ltx23_get_dequant_scratch(min(bytes_per_chunk, n_bytes),
        packed$device)

    start <- 1L
    torch::with_no_grad({
        while (start <= n_bytes) {
            len <- min(bytes_per_chunk, n_bytes - start + 1L)
            chunk <- packed$narrow(1L, start, len)

            # Fully in-place nibble unpack into persistent scratch:
            # hi = byte %/% 16, lo = byte - 16 * hi
            hi <- scratch$hi$narrow(1L, 1L, len)
            lo <- scratch$lo$narrow(1L, 1L, len)
            hi$copy_(chunk)$div_(16L, rounding_mode = "floor")
            lo$copy_(hi)$mul_(-16L)$add_(chunk)

            # Interleave into the (1-based) int64 index scratch
            idx <- scratch$idx$narrow(1L, 1L, len * 2L)
            idx_pairs <- idx$view(c(-1L, 2L))
            idx_pairs$narrow(2L, 1L, 1L)$squeeze(2L)$copy_(hi)
            idx_pairs$narrow(2L, 2L, 1L)$squeeze(2L)$copy_(lo)
            idx$add_(1L)

            vals <- scratch$vals$narrow(1L, 1L, len * 2L)
            .ltx23_index_select_into(vals, scratch$table, idx)

            block_start <- ((start - 1L) * 2L) %/% block + 1L
            n_blocks <- (len * 2L) %/% block
            scales <- absmax$narrow(1L, block_start, n_blocks)
            vals$view(c(-1L, block))$mul_(scales$unsqueeze(2L))

            out_flat$narrow(1L, (start - 1L) * 2L + 1L, len * 2L)$copy_(vals)
            start <- start + len
        }
    })
    out
}

# torch_index_select_out is not exported from torch; fall back to an
# allocating index_select if it ever disappears
.ltx23_index_select_into <- local({
    fn <- NULL
    function(out, table, idx) {
        if (is.null(fn)) {
            fn <<- tryCatch(
                            get("torch_index_select_out", envir = asNamespace("torch")),
                            error = function(e) FALSE
            )
        }
        if (isFALSE(fn)) {
            out$copy_(torch::torch_index_select(table, 1L, idx))
        } else {
            fn(out, table, 1L, idx)
        }
        invisible(out)
    }
})

# Persistent per-device dequantization scratch (nibbles, indices, values,
# and the level table), sized to the chunk length
.ltx23_dequant_scratch <- new.env(parent = emptyenv())

.ltx23_get_dequant_scratch <- function(n_bytes, device) {
    key <- paste(device$type, device$index %||% 0L, sep = "|")
    scratch <- .ltx23_dequant_scratch[[key]]
    if (is.null(scratch) || scratch$n_bytes < n_bytes) {
        scratch <- list(
                        n_bytes = n_bytes,
                        hi = torch::torch_empty(n_bytes, dtype = torch::torch_uint8(),
                device = device),
                        lo = torch::torch_empty(n_bytes, dtype = torch::torch_uint8(),
                device = device),
                        idx = torch::torch_empty(n_bytes * 2L,
                dtype = torch::torch_long(),
                device = device),
                        vals = torch::torch_empty(n_bytes * 2L,
                dtype = torch::torch_float32(),
                device = device),
                        table = torch::torch_tensor(.ltx23_nf4_table,
                dtype = torch::torch_float32(),
                device = device)
        )
        .ltx23_dequant_scratch[[key]] <- scratch
    }
    scratch
}

# Reusable dequantization buffers, keyed by shape/dtype/device: each
# distinct weight shape gets one long-lived buffer, so per-step
# dequantization allocates nothing (the buffer is overwritten in place
# by the next linear of the same shape)
.ltx23_dequant_buffers <- new.env(parent = emptyenv())

.ltx23_get_dequant_buffer <- function(shape, dtype, device) {
    key <- paste(paste(shape, collapse = "x"), dtype$.type(),
                 paste(device$type, device$index %||% 0L), sep = "|")
    buf <- .ltx23_dequant_buffers[[key]]
    if (is.null(buf)) {
        buf <- torch::torch_empty(shape, dtype = dtype, device = device)
        .ltx23_dequant_buffers[[key]] <- buf
    }
    buf
}

#' Release the NF4 dequantization buffers
#'
#' Frees the cached per-shape weight buffers (e.g. before decoding at
#' high resolution).
#'
#' @return Invisibly, NULL.
#'
#' @export
ltx23_release_dequant_buffers <- function() {
    rm(list = ls(.ltx23_dequant_buffers), envir = .ltx23_dequant_buffers)
    rm(list = ls(.ltx23_dequant_scratch), envir = .ltx23_dequant_scratch)
    .ltx23_release_attn_buffers()
    gc(verbose = FALSE)
    if (torch::cuda_is_available()) {
        tryCatch(torch::cuda_empty_cache(), error = function(e) NULL)
    }
    invisible(NULL)
}

#' NF4 linear layer
#'
#' Packed weights and per-block scales are registered as buffers, so
#' they move with the module (uint8 packs are untouched by dtype
#' conversions). The forward pass dequantizes on the weight's device.
#'
#' @param out_features,in_features Integers.
#' @param bias Logical.
#'
#' @export
ltx23_nf4_linear <- torch::nn_module(
                                     "ltx23_nf4_linear",
                                     initialize = function(out_features, in_features, bias = TRUE) {
    self$out_features <- as.integer(out_features)
    self$in_features <- as.integer(in_features)
    n <- self$out_features * self$in_features
    self$weight_nf4 <- torch::nn_buffer(
                                        torch::torch_zeros(n %/% 2L, dtype = torch::torch_uint8())
    )
    self$weight_absmax <- torch::nn_buffer(
        torch::torch_ones(n %/% .ltx23_nf4_block_size,
                          dtype = torch::torch_float32())
    )
    if (bias) {
        self$bias <- torch::nn_parameter(torch::torch_zeros(out_features))
    }
},
                                     set_nf4_weight = function(packed, absmax) {
    torch::with_no_grad({
        self$weight_nf4$copy_(packed)
        self$weight_absmax$copy_(absmax)
    })
    invisible(self)
},
                                     forward = function(x) {
    w <- .ltx23_get_dequant_buffer(
                                   c(self$out_features, self$in_features), x$dtype,
                                   self$weight_nf4$device
    )
    ltx23_nf4_dequantize(
                         self$weight_nf4, self$weight_absmax,
                         c(self$out_features, self$in_features),
                         dtype = x$dtype, out = w
    )
    torch::nnf_linear(x, w, self$bias)
}
)

#' Quantize an LTX-2.3 checkpoint to NF4 shards
#'
#' Same streaming layout and cast policy as
#' \code{\link{ltx23_quantize_fp8}}, but cast-set weights are stored as
#' NF4 (\code{<key>} packed uint8 + \code{<key>_absmax} float32 blocks +
#' the original shape recovered from the model config at load time).
#' Non-cast tensors are copied through unchanged. The manifest carries
#' \code{format = "nf4"}.
#'
#' @param checkpoint_path Source .safetensors (bf16 single file).
#' @param output_dir Output directory for shards + manifest.
#' @param shard_bytes Numeric. Approximate shard size.
#' @param force Logical. Re-quantize even if a valid manifest exists.
#' @param verbose Logical.
#'
#' @return Invisibly, the manifest list.
#'
#' @export
ltx23_quantize_nf4 <- function(checkpoint_path,
                               output_dir = file.path(tools::R_user_dir("diffuseR", "data"), "ltx2.3-nf4"),
                               shard_bytes = 4e9, force = FALSE,
                               verbose = TRUE) {
    manifest_path <- file.path(output_dir, "manifest.json")
    if (!force && file.exists(manifest_path)) {
        manifest <- jsonlite::fromJSON(manifest_path)
        if (all(file.exists(file.path(output_dir, manifest$shards)))) {
            if (verbose) {
                message("NF4 artifact already present: ", output_dir)
            }
            return(invisible(manifest))
        }
    }
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

    ckpt <- ltx23_open_checkpoint(checkpoint_path)

    shard <- list()
    shard_size <- 0
    shard_files <- character(0)
    n_cast <- 0L

    flush_shard <- function() {
        if (!length(shard)) {
            return()
        }
        fname <- sprintf("ltx2.3-nf4-%05d.safetensors",
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

    keys <- ckpt$keys
    for (i in seq_along(keys)) {
        key <- keys[[i]]
        tensor <- ckpt$handle$get_tensor(key)

        mapped <- ltx23_map_dit_key(key)
        if (startsWith(key, "model.diffusion_model.") &&
            ltx23_is_fp8_cast_key(mapped)) {
            torch::with_no_grad({
                q <- ltx23_nf4_quantize(tensor)
                shard[[key]] <- q$packed
                shard[[paste0(key, "_absmax")]] <- q$absmax
            })
            shard_size <- shard_size + prod(tensor$shape) * 0.5625
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
                     source = basename(checkpoint_path),
                     model_version = ckpt$version,
                     format = "nf4",
                     shards = shard_files,
                     tensors = length(keys),
                     nf4_cast = n_cast,
                     config = ckpt$config
    )
    jsonlite::write_json(manifest, manifest_path, auto_unbox = TRUE, pretty = TRUE)
    if (verbose) {
        message(sprintf("Quantized %d/%d tensors to nf4 across %d shards: %s",
                        n_cast, length(keys), length(shard_files), output_dir))
    }
    invisible(manifest)
}

#' Load the LTX-2.3 transformer with resident NF4 weights
#'
#' Builds the transformer, swaps the cast-set linears for
#' \code{\link{ltx23_nf4_linear}}, and loads everything onto
#' \code{device}: at ~4.5 bits/parameter the whole 22B transformer stays
#' GPU-resident, avoiding per-step weight transfers.
#'
#' @param ckpt An NF4 \code{ltx23_checkpoint}
#'   (\code{\link{ltx23_open_fp8_checkpoint}} reads any shard artifact).
#' @param device Character.
#' @param verbose Logical.
#' @param ... Passed to \code{\link{ltx23_transformer}} (tiny test configs).
#'
#' @return The loaded \code{ltx23_transformer}.
#'
#' @export
ltx23_load_transformer_nf4 <- function(ckpt, device = "cuda", verbose = TRUE,
                                       ...) {
    stopifnot(inherits(ckpt, "ltx23_checkpoint"))
    model <- ltx23_transformer(...)
    model$to(dtype = torch::torch_bfloat16())

    groups <- ltx23_split_keys(ckpt$keys)
    dit_keys <- groups$dit
    absmax_keys <- dit_keys[endsWith(dit_keys, ".weight_absmax")]
    main_keys <- setdiff(dit_keys, absmax_keys)

    dests <- c(model$named_parameters(), model$named_buffers())
    filled <- character(0)
    unmapped <- character(0)

    torch::with_no_grad({
        for (i in seq_along(main_keys)) {
            key <- main_keys[[i]]
            mapped <- ltx23_map_dit_key(key)

            if (ltx23_is_fp8_cast_key(mapped) &&
                        paste0(key, "_absmax") %in% absmax_keys) {
                segments <- strsplit(mapped, ".", fixed = TRUE)[[1]]
                parent <- .ltx23_walk_module(model, utils::head(segments, -2L))
                leaf <- segments[length(segments) - 1L]
                old <- .ltx23_walk_module(parent, leaf)
                if (is.null(old)) {
                    unmapped <- c(unmapped, key)
                    next
                }
                w_shape <- old$weight$shape
                nf4_mod <- ltx23_nf4_linear(w_shape[1], w_shape[2],
                    bias = !is.null(old$bias))
                if (!is.null(old$bias)) {
                    # Adopt the original bias parameter; its checkpoint key
                    # loads through the pre-swap destination map
                    nf4_mod$bias <- old$bias
                }
                nf4_mod$set_nf4_weight(
                                       ckpt$handle$get_tensor(key),
                                       ckpt$handle$get_tensor(paste0(key, "_absmax"))
                )
                do.call(`$<-`, list(parent, leaf, nf4_mod))
                filled <- c(filled, mapped)
            } else {
                dest <- dests[[mapped]]
                if (is.null(dest)) {
                    unmapped <- c(unmapped, key)
                    next
                }
                dest$copy_(ckpt$handle$get_tensor(key))
                filled <- c(filled, mapped)
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
        stop("NF4 transformer load: ", length(unmapped), " unmapped keys, e.g. ",
             paste(utils::head(unmapped, 3), collapse = ", "))
    }
    expected_missing <- grepl(.ltx23_fp8_cast_pattern, names(dests))
    unfilled <- setdiff(names(dests)[!expected_missing], filled)
    if (length(unfilled)) {
        stop("NF4 transformer load: ", length(unfilled), " unfilled params, e.g. ",
             paste(utils::head(unfilled, 3), collapse = ", "))
    }

    # Everything (packed weights included, as buffers) onto the GPU
    model$to(device = device)
    model$eval()
    # Block intermediates (norms, projections, FF activations) are still
    # ~1.5GB per block at high resolution; per-block gc keeps them bounded
    options(diffuseR.block_gc = TRUE)
    if (verbose) {
        message("Transformer ready: NF4 weights resident on ", device)
    }
    model
}
