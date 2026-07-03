#' FP8 Weight Storage for the LTX-2.3 Transformer
#'
#' GPU-poor weight handling: the large attention/FFN linears of the DiT
#' are stored as float8_e4m3fn with per-tensor scales (the official LTX
#' quantization policy), kept CPU-resident (optionally pinned), and
#' dequantized on the compute device inside each forward. Everything
#' else (norms, embeddings, modulation tables, biases) stays bfloat16.
#' Requires a safetensors build with F8 dtype support.
#'
#' @name fp8_ltx23
NULL

# Official LTX fp8 cast policy: attention and FFN linear weights inside
# the transformer blocks (the dotless ff suffixes also catch audio_ff)
.ltx23_fp8_cast_pattern <- paste0(
  "^transformer_blocks\\.[0-9]+\\..*",
  "(to_q|to_k|to_v|to_out\\.0|ff\\.net\\.0\\.proj|ff\\.net\\.2)\\.weight$"
)

#' Test whether a mapped DiT key is in the official fp8 cast set
#'
#' @param mapped_key Character vector of mapped (diffusers-style)
#'   parameter names.
#'
#' @return Logical vector.
#'
#' @export
ltx23_is_fp8_cast_key <- function(mapped_key) {
  grepl(.ltx23_fp8_cast_pattern, mapped_key)
}

#' FP8 linear layer
#'
#' Weight lives as float8_e4m3fn plus a float32 scale in plain module
#' fields (so \code{$to(device)} moves only the bias); the forward pass
#' ships 1 byte/param to the input's device, upcasts, rescales, and runs
#' \code{nnf_linear}.
#'
#' @param out_features,in_features Integers.
#' @param bias Logical.
#'
#' @export
ltx23_fp8_linear <- torch::nn_module(
  "ltx23_fp8_linear",
  initialize = function(out_features, in_features, bias = TRUE) {
    self$out_features <- as.integer(out_features)
    self$in_features <- as.integer(in_features)
    self$weight_fp8 <- NULL
    self$weight_scale <- NULL
    if (bias) {
      self$bias <- torch::nn_parameter(torch::torch_zeros(out_features))
    }
  },
  set_fp8_weight = function(weight, scale, pin = FALSE) {
    weight <- weight$to(device = "cpu")
    if (pin && torch::cuda_is_available()) {
      weight <- weight$pin_memory(device = torch::torch_device("cuda"))
    }
    self$weight_fp8 <- weight
    self$weight_scale <- scale$to(device = "cpu", dtype = torch::torch_float32())
    invisible(self)
  },
  forward = function(x) {
    # Transfer fp8 bytes first, cast on the compute device second
    w <- self$weight_fp8$to(device = x$device, non_blocking = TRUE)
    w <- w$to(dtype = x$dtype) * self$weight_scale$to(device = x$device, dtype = x$dtype)
    out <- torch::nnf_linear(x, w, self$bias)
    rm(w)
    out
  }
)

# Navigate a dotted parameter path to a submodule. Numeric segments
# index module lists (0-based names -> 1-based R); named segments go
# through the nn_module `$` accessor.
.ltx23_walk_module <- function(module, segments) {
  cur <- module
  for (seg in segments) {
    cur <- if (grepl("^[0-9]+$", seg)) {
      cur[[as.integer(seg) + 1L]]
    } else {
      do.call(`$`, list(cur, seg))
    }
    if (is.null(cur)) return(NULL)
  }
  cur
}

#' Quantize an LTX-2.3 checkpoint to FP8 shards
#'
#' Streams the single-file bf16 checkpoint tensor by tensor. DiT
#' attention/FFN linear weights are stored as float8_e4m3fn with a
#' float32 absmax/448 per-tensor scale (\code{<key>_scale} sibling);
#' everything else is copied through unchanged. Output shards carry the
#' original key names plus a manifest for skip-if-exists.
#'
#' @param checkpoint_path Source .safetensors (46 GB bf16 single file).
#' @param output_dir Output directory for shards + manifest.
#' @param shard_bytes Numeric. Approximate shard size (default 4 GB).
#' @param force Logical. Re-quantize even if a valid manifest exists.
#' @param verbose Logical.
#'
#' @return Invisibly, the manifest list.
#'
#' @export
ltx23_quantize_fp8 <- function(
  checkpoint_path,
  output_dir = file.path(tools::R_user_dir("diffuseR", "data"), "ltx2.3-fp8"),
  shard_bytes = 4e9,
  force = FALSE,
  verbose = TRUE
) {
  manifest_path <- file.path(output_dir, "manifest.json")
  if (!force && file.exists(manifest_path)) {
    manifest <- jsonlite::fromJSON(manifest_path)
    if (all(file.exists(file.path(output_dir, manifest$shards)))) {
      if (verbose) message("FP8 artifact already present: ", output_dir)
      return(invisible(manifest))
    }
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  ckpt <- ltx23_open_checkpoint(checkpoint_path)
  fp8 <- torch::torch_float8_e4m3fn()

  shard <- list()
  shard_size <- 0
  shard_files <- character(0)
  n_cast <- 0L

  flush_shard <- function() {
    if (!length(shard)) return()
    fname <- sprintf("ltx2.3-fp8-%05d.safetensors", length(shard_files) + 1L)
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

    mapped <- ltx23_map_dit_key(key)
    if (startsWith(key, "model.diffusion_model.") && ltx23_is_fp8_cast_key(mapped)) {
      torch::with_no_grad({
        scale <- tensor$abs()$max()$to(dtype = torch::torch_float32())$
          clamp(min = 1e-12) / 448
        shard[[key]] <- (tensor$to(dtype = torch::torch_float32()) / scale)$to(dtype = fp8)
        shard[[paste0(key, "_scale")]] <- scale
      })
      shard_size <- shard_size + prod(tensor$shape)
      n_cast <- n_cast + 1L
    } else {
      shard[[key]] <- tensor
      shard_size <- shard_size + prod(tensor$shape) * 2
    }
    rm(tensor)

    if (shard_size >= shard_bytes) flush_shard()
    if (i %% 200L == 0L) {
      gc(verbose = FALSE)
      if (verbose) message(sprintf("  quantizing %d/%d tensors", i, length(keys)))
    }
  }
  flush_shard()

  manifest <- list(
    source = basename(checkpoint_path),
    model_version = ckpt$version,
    shards = shard_files,
    tensors = length(keys),
    fp8_cast = n_cast,
    config = ckpt$config
  )
  jsonlite::write_json(manifest, manifest_path, auto_unbox = TRUE, pretty = TRUE)
  if (verbose) {
    message(sprintf("Quantized %d/%d tensors to fp8 across %d shards: %s",
      n_cast, length(keys), length(shard_files), output_dir))
  }
  invisible(manifest)
}

#' Open an FP8 shard directory as a checkpoint
#'
#' Presents the sharded fp8 artifact through the same interface as
#' \code{\link{ltx23_open_checkpoint}} so the group loaders work
#' unchanged.
#'
#' @param dir The fp8 artifact directory (with manifest.json).
#'
#' @return An \code{ltx23_checkpoint}.
#'
#' @export
ltx23_open_fp8_checkpoint <- function(dir) {
  manifest_path <- file.path(dir, "manifest.json")
  if (!file.exists(manifest_path)) {
    stop("No manifest.json in ", dir, "; run ltx23_quantize_fp8() first.")
  }
  manifest <- jsonlite::fromJSON(manifest_path, simplifyVector = TRUE)

  handles <- lapply(file.path(dir, manifest$shards), function(p) {
    safetensors::safetensors$new(p, framework = "torch")
  })
  key_to_handle <- list()
  for (h in handles) {
    for (k in setdiff(h$keys(), "__metadata__")) key_to_handle[[k]] <- h
  }

  handle <- list(
    get_tensor = function(key) {
      h <- key_to_handle[[key]]
      if (is.null(h)) stop("Key not found in fp8 shards: ", key)
      h$get_tensor(key)
    }
  )

  structure(
    list(
      handle = handle,
      keys = names(key_to_handle),
      version = manifest$model_version,
      config = manifest$config,
      path = dir
    ),
    class = "ltx23_checkpoint"
  )
}

#' Load the LTX-2.3 transformer with FP8 weights
#'
#' Builds the transformer, swaps the official cast-set linears for
#' \code{\link{ltx23_fp8_linear}}, loads fp8 weights CPU-side (optionally
#' pinned) and everything else as bfloat16 on \code{device}. Sets
#' \code{options(diffuseR.use_fp8 = TRUE)} so the transformer runs
#' per-block garbage collection over the dequantized temporaries.
#'
#' @param ckpt An fp8 \code{ltx23_checkpoint}
#'   (\code{\link{ltx23_open_fp8_checkpoint}}).
#' @param device Character. Device for the resident (non-fp8) weights.
#' @param pin Logical. Pin the fp8 host memory for faster transfers.
#' @param verbose Logical.
#' @param ... Passed to \code{\link{ltx23_transformer}} (tiny test configs).
#'
#' @return The loaded \code{ltx23_transformer}.
#'
#' @export
ltx23_load_transformer_fp8 <- function(
  ckpt,
  device = "cuda",
  pin = TRUE,
  verbose = TRUE,
  ...
) {
  stopifnot(inherits(ckpt, "ltx23_checkpoint"))
  model <- ltx23_transformer(...)
  model$to(dtype = torch::torch_bfloat16())

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

      if (ltx23_is_fp8_cast_key(mapped) && paste0(key, "_scale") %in% scale_keys) {
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
        fp8_mod <- ltx23_fp8_linear(
          weight$shape[1], weight$shape[2],
          bias = !is.null(old$bias)
        )
        if (!is.null(old$bias)) {
          # Adopt the original bias parameter so the separate bias key,
          # which copies through the pre-swap destination map, lands here
          fp8_mod$bias <- old$bias
        }
        fp8_mod$set_fp8_weight(weight, scale, pin = pin)
        do.call(`$<-`, list(parent, leaf, fp8_mod))
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
          message(sprintf("  loaded %d/%d transformer tensors", i, length(main_keys)))
        }
      }
    }
  })
  gc(verbose = FALSE)

  if (length(unmapped)) {
    stop("FP8 transformer load: ", length(unmapped), " unmapped keys, e.g. ",
      paste(utils::head(unmapped, 3), collapse = ", "))
  }

  # Weight params replaced by fp8 modules won't be "filled"; account for them
  expected_missing <- grepl(.ltx23_fp8_cast_pattern, names(dests))
  unfilled <- setdiff(names(dests)[!expected_missing], filled)
  if (length(unfilled)) {
    stop("FP8 transformer load: ", length(unfilled), " unfilled params, e.g. ",
      paste(utils::head(unfilled, 3), collapse = ", "))
  }

  # Residents (norms, embeddings, tables, biases) to the compute device;
  # fp8 fields stay on the CPU because they are plain fields
  model$to(device = device)
  model$eval()
  options(diffuseR.use_fp8 = TRUE)
  if (verbose) message("Transformer ready: fp8 weights CPU-resident, rest on ", device)
  model
}

#' Set the attention query-chunk size across a transformer
#'
#' R torch has no fused attention, so the [B, H, S, S] matrix
#' materializes; chunking queries bounds the peak. NULL disables
#' chunking.
#'
#' @param transformer An \code{ltx23_transformer}.
#' @param chunk Integer or NULL.
#'
#' @return Invisibly, the transformer.
#'
#' @export
ltx23_set_attn_chunk <- function(transformer, chunk) {
  for (i in seq_along(transformer$transformer_blocks)) {
    block <- transformer$transformer_blocks[[i]]
    for (name in c("attn1", "audio_attn1", "attn2", "audio_attn2",
      "audio_to_video_attn", "video_to_audio_attn")) {
      block[[name]]$attn_chunk <- chunk
    }
  }
  invisible(transformer)
}
