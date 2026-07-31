## STATUS 2026-07-31: benchmarked against NF4 on a 16 GB card and it is
## 3.27x SLOWER end to end (105.6 s vs 32.3 s warm, 768x512x49), even
## though the quantizer itself decodes 5.2x faster with 4.3x lower error.
## Cause: this file never touches the jit path, so ~3,700 cast-set
## weights run as eager R forwards every step. Also, the int8 cast set is
## 18.52 GB and cannot be GPU-resident on 16 GB. Not shippable as-is.
## See INT8_STATUS.md at the repo root for the numbers, the VRAM
## arithmetic, and the jit_trace constant-folding landmine that makes a
## naive fusion attempt silently stop streaming.

#' INT8 Weight Storage for the LTX-2.3 Transformer
#'
#' Streamed 8-bit weights: the cast-set linears (the official LTX fp8
#' cast policy) are stored as int8 with per-output-channel float32
#' scales, kept CPU-resident (optionally pinned), and shipped to the
#' compute device inside each forward - 1 byte/param over PCIe, upcast
#' and rescale on device. Per-channel scales carry more precision than
#' fp8's per-tensor scale, and int8 is a standard safetensors dtype, so
#' the artifact reads on stock CRAN safetensors (fp8 needs the fork).
#'
#' @name int8_ltx23
NULL

#' Quantize a tensor to int8 with per-output-channel scales
#'
#' @param x Float 2D weight [out, in].
#'
#' @return List with \code{weight_int8} (int8 [out, in]) and
#'   \code{scale} (float32 [out], absmax/127 per row).
#'
#' @export
ltx23_int8_quantize <- function(x) {
    stopifnot(x$ndim == 2L)
    w <- x$detach()$to(dtype = torch::torch_float32())
    scale <- w$abs()$amax(dim = 2L)$clamp(min = 1e-12)$div(127)
    wi8 <- torch::torch_round(w / scale$unsqueeze(2L))$
    clamp(-127, 127)$to(dtype = torch::torch_int8())
    list(weight_int8 = wi8, scale = scale)
}

#' INT8 linear layer
#'
#' Weight lives as int8 plus per-channel float32 scales in plain module
#' fields (so \code{$to(device)} moves only the bias); the forward pass
#' ships 1 byte/param to the input's device, upcasts into a persistent
#' per-shape buffer, and rescales in place.
#'
#' @param out_features,in_features Integers.
#' @param bias Logical.
#'
#' @export
ltx23_int8_linear <- torch::nn_module(
                                      "ltx23_int8_linear",
                                      initialize = function(out_features, in_features, bias = TRUE) {
    self$out_features <- as.integer(out_features)
    self$in_features <- as.integer(in_features)
    self$weight_int8 <- NULL
    self$weight_scale <- NULL
    if (bias) {
        self$bias <- torch::nn_parameter(torch::torch_zeros(out_features))
    }
},
                                      set_int8_weight = function(weight, scale, pin = FALSE) {
    weight <- weight$to(device = "cpu")
    if (pin && torch::cuda_is_available()) {
        weight <- weight$pin_memory(device = torch::torch_device("cuda"))
    }
    self$weight_int8 <- weight
    self$weight_scale <- scale$to(device = "cpu",
                                  dtype = torch::torch_float32())
    invisible(self)
},
                                      forward = function(x) {
    # Transfer int8 bytes first (no-op when resident), then cast into a
    # persistent per-shape buffer and scale in place - zero fresh
    # allocations per call. The scale is cast down before the multiply
    # so nothing promotes to float32.
    w8 <- self$weight_int8$to(device = x$device, non_blocking = TRUE)
    w <- .ltx23_get_dequant_buffer(
                                   c(self$out_features, self$in_features), x$dtype, x$device
    )
    torch::with_no_grad({
        w$copy_(w8)
        w$mul_(self$weight_scale$to(device = x$device,
                                    dtype = x$dtype)$unsqueeze(2L))
    })
    torch::nnf_linear(x, w, self$bias)
}
)

#' Quantize an LTX-2.3 checkpoint to int8 shards
#'
#' Same streaming layout and cast policy as
#' \code{\link{ltx23_quantize_fp8}}, but cast-set weights are stored as
#' int8 with per-output-channel \code{<key>_scale} float32 vectors.
#' Non-cast tensors are copied through unchanged. The manifest carries
#' \code{format = "int8"}. No special safetensors build is required.
#'
#' @param checkpoint_path Source .safetensors (bf16 single file).
#' @param output_dir Output directory for shards + manifest.
#' @param shard_bytes Numeric. Target shard size in bytes (the 1.9e9
#'   default keeps shards readable by stock CRAN safetensors).
#' @param force Logical. Re-quantize even if a valid manifest exists.
#' @param verbose Logical.
#'
#' @return Invisibly, the manifest list.
#'
#' @export
ltx23_quantize_int8 <- function(checkpoint_path,
                                output_dir = file.path(tools::R_user_dir("diffuseR", "data"), "ltx2.3-int8"),
                                shard_bytes = 1.9e9, force = FALSE,
                                verbose = TRUE) {
    manifest_path <- file.path(output_dir, "manifest.json")
    if (!force && file.exists(manifest_path)) {
        manifest <- jsonlite::fromJSON(manifest_path)
        if (identical(manifest$format, "int8") &&
            all(file.exists(file.path(output_dir, manifest$shards)))) {
            if (verbose) {
                message("INT8 artifact already present: ", output_dir)
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
        fname <- sprintf("ltx2.3-int8-%05d.safetensors",
                         length(shard_files) + 1L)
        safetensors::safe_save_file(shard, file.path(output_dir, fname))
        shard_files[[length(shard_files) + 1L]] <<- fname
        if (verbose) {
            message(sprintf("  wrote %s (%.2f GB, %d tensors)",
                            fname, shard_size / 1e9, length(shard)))
        }
        shard <<- list()
        shard_size <<- 0
        gc(verbose = FALSE)
    }

    keys <- ckpt$keys
    for (i in seq_along(keys)) {
        key <- keys[[i]]
        tensor <- ckpt$handle$get_tensor(key)

        # Flush BEFORE adding when this tensor would push the shard
        # over target: flushing after (the fp8/nf4 quantizers' pattern)
        # can emit a shard up to one tensor over shard_bytes, past the
        # 2^31 ceiling stock CRAN safetensors can read back
        this_bytes <- prod(tensor$shape) *
        if (startsWith(key, "model.diffusion_model.") &&
                ltx23_is_fp8_cast_key(ltx23_map_dit_key(key))) 1 else 2
        if (length(shard) && shard_size + this_bytes > shard_bytes) {
            flush_shard()
        }

        mapped <- ltx23_map_dit_key(key)
        if (startsWith(key, "model.diffusion_model.") &&
            ltx23_is_fp8_cast_key(mapped)) {
            torch::with_no_grad({
                q <- ltx23_int8_quantize(tensor)
                shard[[key]] <- q$weight_int8
                shard[[paste0(key, "_scale")]] <- q$scale
            })
            shard_size <- shard_size + prod(tensor$shape)
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
                     format = "int8",
                     shards = shard_files,
                     tensors = length(keys),
                     int8_cast = n_cast,
                     config = ckpt$config
    )
    jsonlite::write_json(manifest, manifest_path, auto_unbox = TRUE, pretty = TRUE)
    if (verbose) {
        message(sprintf("Quantized %d/%d tensors to int8 across %d shards: %s",
                        n_cast, length(keys), length(shard_files), output_dir))
    }
    invisible(manifest)
}

#' Load the LTX-2.3 transformer from an int8 artifact
#'
#' Cast-set linears become \code{\link{ltx23_int8_linear}} modules
#' (int8 host-resident, optionally pinned, streamed per forward);
#' everything else loads as bfloat16 on \code{device}. Hard-errors on
#' unmapped or unfilled parameters.
#'
#' @param ckpt An \code{ltx23_checkpoint} from
#'   \code{\link{ltx23_open_fp8_checkpoint}} on an int8 artifact.
#' @param device "cuda" or "cpu".
#' @param pin Logical. Pin the int8 host memory for full-rate DMA.
#' @param verbose Logical.
#' @param ... Passed to \code{ltx23_transformer}.
#'
#' @return The transformer module.
#'
#' @export
ltx23_load_transformer_int8 <- function(ckpt, device = "cuda", pin = TRUE,
                                        verbose = TRUE, ...) {
    stopifnot(inherits(ckpt, "ltx23_checkpoint"))
    model <- .construct_skeleton(ltx23_transformer, ...)

    groups <- ltx23_split_keys(ckpt$keys)
    dit_keys <- groups$dit
    scale_keys <- dit_keys[endsWith(dit_keys, ".weight_scale")]
    main_keys <- setdiff(dit_keys, scale_keys)

    dests <- c(model$named_parameters(), model$named_buffers())
    filled <- character(0)
    unmapped <- character(0)

    torch::with_no_grad({
        for (i in seq_along(main_keys)) {
            key <- main_keys[[i]]
            mapped <- ltx23_map_dit_key(key)

            if (ltx23_is_fp8_cast_key(mapped) &&
                        paste0(key, "_scale") %in% scale_keys) {
                segments <- strsplit(mapped, ".", fixed = TRUE)[[1]]
                parent <- .ltx23_walk_module(model, utils::head(segments, -2L))
                leaf <- segments[length(segments) - 1L]
                old <- .ltx23_walk_module(parent, leaf)
                if (is.null(old)) {
                    unmapped <- c(unmapped, key)
                    next
                }
                weight <- ckpt$handle$get_tensor(key)
                scale <- ckpt$handle$get_tensor(paste0(key, "_scale"))
                int8_mod <- ltx23_int8_linear(weight$shape[1], weight$shape[2],
                    bias = !is.null(old$bias))
                if (!is.null(old$bias)) {
                    # Adopt the original bias parameter so the separate
                    # bias key, which copies through the pre-swap
                    # destination map, lands here
                    int8_mod$bias <- old$bias
                }
                int8_mod$set_int8_weight(weight, scale, pin = pin)
                do.call(`$<-`, list(parent, leaf, int8_mod))
                filled <- c(filled, mapped)
                rm(weight, scale)
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
        stop("INT8 transformer load: ", length(unmapped),
             " unmapped keys, e.g. ",
             paste(utils::head(unmapped, 3), collapse = ", "))
    }
    expected_missing <- grepl(.ltx23_fp8_cast_pattern, names(dests))
    unfilled <- setdiff(names(dests)[!expected_missing], filled)
    if (length(unfilled)) {
        stop("INT8 transformer load: ", length(unfilled),
             " unfilled params, e.g. ",
             paste(utils::head(unfilled, 3), collapse = ", "))
    }

    model$to(device = device)
    model$eval()
    if (verbose) {
        message("Transformer ready: int8 weights host-resident",
                if (pin) " (pinned)" else "", ", streamed per forward")
    }
    model
}