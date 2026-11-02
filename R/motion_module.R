#' AnimateDiff motion modules
#'
#' Temporal transformers that turn the SD 1.5 image UNet into a video
#' model: after every resnet (and spatial attention) the UNet runs one
#' motion module, which attends across frames at each latent position.
#' Ported from diffusers \code{models/unets/unet_motion_model.py}
#' (\code{AnimateDiffTransformer3D}, \code{MotionModules},
#' \code{MotionAdapter}, \code{UNetMotionModel}) and the double
#' self-attention \code{BasicTransformerBlock} in
#' \code{models/attention.py}, Apache-2.0.
#'
#' @name motion_module
#' @keywords internal
NULL

# Sinusoidal frame positions [1, max_len, dim]: sin on even channels, cos
# on odd (diffusers SinusoidalPositionalEmbedding). Recomputed rather than
# loaded; the checkpoints' pos_encoder buffers hold the same table.
.motion_positions <- function(dim, max_len) {
    pos <- seq.int(0L, max_len - 1L)
    div <- exp(seq(0, dim - 1, by = 2) * (-log(10000) / dim))
    ang <- outer(pos, div)
    pe <- matrix(0, max_len, dim)
    pe[, seq(1L, dim, by = 2L)] <- sin(ang)
    pe[, seq(2L, dim, by = 2L)] <- cos(ang)
    torch::torch_tensor(pe, dtype = torch::torch_float32())$unsqueeze(1L)
}

# One temporal transformer layer: two self-attentions over the frame axis
# (positions added to each normed input) and a GEGLU feed-forward.
motion_transformer_block <- torch::nn_module(
    "MotionTransformerBlock",
    initialize = function(dim, heads, max_len) {
        self$norm1 <- torch::nn_layer_norm(dim)
        self$attn1 <- UNetCrossAttention(dim, heads = heads,
                                         dim_head = dim %/% heads)
        self$norm2 <- torch::nn_layer_norm(dim)
        self$attn2 <- UNetCrossAttention(dim, heads = heads,
                                         dim_head = dim %/% heads)
        self$norm3 <- torch::nn_layer_norm(dim)
        self$ff <- FeedForward(dim)
        self$pe <- torch::nn_buffer(.motion_positions(dim, max_len),
                                    persistent = FALSE)
    },
    attend = function(x) {
        pe <- self$pe[, seq_len(x$shape[2]), ]$to(dtype = x$dtype)
        x <- x + self$attn1(self$norm1(x) + pe)
        x + self$attn2(self$norm2(x) + pe)
    },
    forward = function(x, windows = NULL) {
        if (is.null(windows)) {
            x <- self$attend(x)
        } else {
            x <- .motion_windowed(x, windows, self$attend)
        }
        x + self$ff(self$norm3(x))
    }
)

# FreeNoise (diffusers FreeNoiseTransformerBlock, Apache-2.0): the two
# attentions run on overlapping frame windows, each with positions from
# 0, and each frame takes the weighted mean of its windows' outputs. The
# feed-forward is per frame, so it runs once on the merged result.
.motion_windowed <- function(x, windows, attend) {
    n <- x$shape[2]
    acc <- torch::torch_zeros(c(x$shape[1], n, x$shape[3]), device = x$device)
    cnt <- torch::torch_zeros(c(1L, n, 1L), device = x$device)
    for (w in windows$frames) {
        len <- w$end - w$start + 1L
        h <- attend(x$narrow(2L, w$start, len))$to(dtype = torch::torch_float32())
        k1 <- w$keep[1]
        nk <- length(w$keep)
        wt <- torch::torch_tensor(windows$weights[w$keep], device = x$device)$
            view(c(1L, -1L, 1L))
        # narrow() views: indexing with a stored range copies, and add_ on
        # the copy would leave acc untouched
        acc$narrow(2L, w$start - 1L + k1, nk)$add_(h$narrow(2L, k1, nk) * wt)
        cnt$narrow(2L, w$start - 1L + k1, nk)$add_(wt)
    }
    (acc / cnt)$to(dtype = x$dtype)
}

# Frame windows for FreeNoise: length `len` every `stride` frames, plus
# one window ending on the last frame when the stride does not land
# there; that window contributes only the frames no earlier window
# covered. diffusers counts those frames with a single scalar weight
# (`weights[:, -n]`) while summing them with per-frame weights; here
# both use the per-frame weights, so a proper weighted mean.
.motion_windows <- function(num_frames, len = 16L, stride = 4L,
                            weighting = c("pyramid", "flat")) {
    weighting <- match.arg(weighting)
    if (num_frames < len) {
        stop(sprintf("%d frames are fewer than the %d-frame context window",
                     num_frames, len), call. = FALSE)
    }
    starts <- seq.int(1L, num_frames - len + 1L, by = stride)
    frames <- lapply(starts, function(s) {
        list(start = s, end = s + len - 1L, keep = seq_len(len))
    })
    last <- starts[length(starts)] + len - 1L
    if (last < num_frames) {
        extra <- num_frames - last
        frames <- c(frames, list(list(start = num_frames - len + 1L,
                                      end = num_frames,
                                      keep = seq.int(len - extra + 1L, len))))
    }
    half <- len %/% 2L
    weights <- if (weighting == "flat") {
        rep(1, len)
    } else if (len %% 2L == 0L) {
        c(seq_len(half), rev(seq_len(half)))
    } else {
        c(seq_len(half), half + 1L, rev(seq_len(half)))
    }
    list(frames = frames, weights = as.numeric(weights))
}

# AnimateDiffTransformer3D: [B*F, C, H, W] in and out. The group norm runs
# over [B, C, F, H, W], so its statistics span all frames.
motion_transformer_3d <- torch::nn_module(
    "MotionTransformer3D",
    initialize = function(channels, heads = 8L, max_len = 32L, layers = 1L) {
        self$norm <- torch::nn_group_norm(32L, channels, eps = 1e-6)
        self$proj_in <- torch::nn_linear(channels, channels)
        self$transformer_blocks <- torch::nn_module_list()
        for (i in seq_len(layers)) {
            self$transformer_blocks$append(
                motion_transformer_block(channels, heads, max_len))
        }
        self$proj_out <- torch::nn_linear(channels, channels)
    },
    forward = function(x, num_frames, windows = NULL) {
        d <- as.integer(x$shape)
        b <- d[1] %/% num_frames
        residual <- x
        x <- x$reshape(c(b, num_frames, d[2], d[3], d[4]))$permute(c(1L, 3L, 2L, 4L, 5L))
        x <- self$norm(x)
        x <- x$permute(c(1L, 4L, 5L, 3L, 2L))$reshape(c(b * d[3] * d[4], num_frames, d[2]))
        x <- self$proj_in(x)
        for (i in seq_along(self$transformer_blocks)) {
            x <- self$transformer_blocks[[i]](x, windows)
        }
        x <- self$proj_out(x)
        x <- x$reshape(c(b, d[3], d[4], num_frames, d[2]))$
            permute(c(1L, 4L, 5L, 2L, 3L))$contiguous()$reshape(d)
        x + residual
    }
)

motion_modules <- torch::nn_module(
    "MotionModules",
    initialize = function(channels, n, heads, max_len) {
        self$motion_modules <- torch::nn_module_list()
        for (i in seq_len(n)) {
            self$motion_modules$append(
                motion_transformer_3d(channels, heads, max_len))
        }
    }
)

#' AnimateDiff motion adapter
#'
#' The motion modules for every block of an SD UNet, laid out as
#' diffusers' \code{MotionAdapter} so its parameter names match the
#' converted checkpoints: \code{down_blocks.<i>.motion_modules.<j>},
#' \code{mid_block.motion_modules.0} and \code{up_blocks.<i>.motion_modules.<j>}.
#' Up blocks carry one more module than down blocks, as the UNet's up
#' blocks carry one more resnet.
#'
#' @param block_out_channels UNet block widths.
#' @param layers_per_block Motion modules per down block.
#' @param use_mid_block Whether the mid block has a motion module (v2
#'   does, v3 does not).
#' @param num_heads Attention heads per motion module.
#' @param max_len Longest frame window the positions cover.
#'
#' @return An \code{nn_module}.
#' @export
motion_adapter <- torch::nn_module(
    "MotionAdapter",
    initialize = function(block_out_channels = c(320L, 640L, 1280L, 1280L),
                          layers_per_block = 2L, use_mid_block = TRUE,
                          num_heads = 8L, max_len = 32L) {
        self$max_len <- max_len
        self$down_blocks <- torch::nn_module_list()
        for (ch in block_out_channels) {
            self$down_blocks$append(
                motion_modules(ch, layers_per_block, num_heads, max_len))
        }
        if (use_mid_block) {
            self$mid_block <- motion_modules(
                block_out_channels[length(block_out_channels)], 1L,
                num_heads, max_len)
        }
        self$up_blocks <- torch::nn_module_list()
        for (ch in rev(block_out_channels)) {
            self$up_blocks$append(
                motion_modules(ch, layers_per_block + 1L, num_heads, max_len))
        }
    }
)

# Original AnimateDiff key -> diffusers MotionAdapter key (the mapping in
# diffusers' scripts/convert_animatediff_motion_module_to_diffusers.py).
# Already-converted keys pass through unchanged.
.motion_remap <- function(key) {
    key <- sub("^module\\.", "", key)
    for (r in list(c(".temporal_transformer", ""), c(".norms.0", ".norm1"),
                   c(".norms.1", ".norm2"), c(".ff_norm", ".norm3"),
                   c(".attention_blocks.0", ".attn1"),
                   c(".attention_blocks.1", ".attn2"))) {
        key <- gsub(r[1], r[2], key, fixed = TRUE)
    }
    key
}

# Adapter shape from its keys: widths from each down block's first norm,
# the mid block's presence, and the frame window from a stored position
# table (32 when none is stored).
.motion_adapter_config <- function(handle, keys) {
    keys <- vapply(keys, .motion_remap, "")
    norm <- grep("^down_blocks\\.[0-9]+\\.motion_modules\\.0\\.norm\\.weight$",
                 keys, value = TRUE)
    if (!length(norm)) {
        stop("Not an AnimateDiff motion module: no down_blocks.*.motion_modules",
             call. = FALSE)
    }
    idx <- as.integer(sub("^down_blocks\\.([0-9]+)\\..*", "\\1", norm))
    norm <- norm[order(idx)]
    raw <- names(keys)[match(norm, keys)]
    widths <- vapply(raw, function(k) as.integer(handle$get_tensor(k)$shape[1]),
                     integer(1), USE.NAMES = FALSE)
    per_block <- length(unique(sub("^down_blocks\\.0\\.motion_modules\\.([0-9]+)\\..*",
        "\\1", grep("^down_blocks\\.0\\.motion_modules\\.", keys, value = TRUE))))
    pe <- grep("pos_encoder\\.pe$|pos_embed\\.pe$", names(keys), value = TRUE)
    max_len <- if (length(pe)) {
        as.integer(handle$get_tensor(pe[1])$shape[2])
    } else {
        32L
    }
    list(block_out_channels = widths, layers_per_block = per_block,
         use_mid_block = any(startsWith(keys, "mid_block.")),
         max_len = max_len)
}

#' Load an AnimateDiff motion adapter from safetensors
#'
#' Reads a motion module checkpoint in either the original AnimateDiff key
#' layout (as converted by \code{\link{download_animatediff}}) or diffusers'
#' \code{MotionAdapter} layout, builds a matching
#' \code{\link{motion_adapter}} and fills every parameter. The stored
#' position tables are skipped (the module computes the same sinusoids);
#' any other unmatched key, shape mismatch or unfilled parameter is an
#' error.
#'
#' @param path A \code{.safetensors} file.
#' @param verbose Logical.
#'
#' @return A \code{\link{motion_adapter}} in eval mode (float32, CPU).
#' @export
load_motion_adapter <- function(path, verbose = TRUE) {
    handle <- safetensors::safetensors$new(path.expand(path), framework = "torch")
    keys <- handle$keys()
    cfg <- .motion_adapter_config(handle, keys)
    adapter <- do.call(motion_adapter, cfg)
    dests <- adapter$named_parameters()
    filled <- character(0)
    torch::with_no_grad({
        for (key in keys) {
            if (grepl("pos_encoder\\.pe$|pos_embed\\.pe$", key)) {
                next
            }
            name <- .motion_remap(key)
            dest <- dests[[name]]
            if (is.null(dest)) {
                stop("Motion adapter load: unmapped key ", key, call. = FALSE)
            }
            src <- handle$get_tensor(key)
            if (!identical(as.integer(src$shape), as.integer(dest$shape))) {
                stop("Motion adapter load: shape mismatch at ", name,
                     call. = FALSE)
            }
            dest$copy_(src)
            filled <- c(filled, name)
        }
    })
    unfilled <- setdiff(names(dests), filled)
    if (length(unfilled)) {
        stop("Motion adapter load: ", length(unfilled), " unfilled, e.g. ",
             paste(utils::head(unfilled, 3), collapse = ", "), call. = FALSE)
    }
    if (verbose) {
        message(sprintf("Loaded %d motion adapter tensors (%s mid block, %d-frame window)",
                        length(filled), if (cfg$use_mid_block) "with" else "no",
                        cfg$max_len))
    }
    adapter$eval()
    adapter
}

#' SD UNet with AnimateDiff motion modules
#'
#' Wraps a native SD 1.5 \code{\link{unet_native}} and a
#' \code{\link{motion_adapter}}. The forward takes latents
#' \code{[B, 4, F, H, W]}; frames run through the image layers as a batch
#' of \code{B * F} and through each motion module along the frame axis.
#' The text context is either \code{[B, 77, D]} (repeated per frame) or
#' \code{[B * F, 77, D]}, one row per frame, which is how prompt travel
#' conditions each frame on its own prompt.
#'
#' The forward also takes \code{context_length} and \code{context_stride}:
#' with more frames than \code{context_length}, the motion modules attend
#' within overlapping windows of that length (FreeNoise) instead of over
#' the whole clip.
#'
#' @param unet A \code{\link{unet_native}}.
#' @param adapter A \code{\link{motion_adapter}} with the same block widths.
#'
#' @return An \code{nn_module}.
#' @export
unet_motion <- torch::nn_module(
    "UNetMotion",
    initialize = function(unet, adapter) {
        self$unet <- unet
        self$adapter <- adapter
    },
    forward = function(sample, timestep, encoder_hidden_states,
                       context_length = NULL, context_stride = 4L) {
        u <- self$unet
        a <- self$adapter
        d <- as.integer(sample$shape)
        f <- d[3]
        windows <- NULL
        if (!is.null(context_length) && f > context_length) {
            windows <- .motion_windows(f, context_length, context_stride)
        } else if (f > a$max_len) {
            stop(sprintf("%d frames exceed the motion module's %d-frame window",
                         f, a$max_len), call. = FALSE)
        }
        # Minor gc per layer (diffuseR.block_gc, as the DiTs): a 16-frame
        # forward leaves GBs of dead activations behind otherwise
        block_gc <- isTRUE(getOption("diffuseR.block_gc"))
        model_dtype <- u$time_embedding_linear_1$weight$dtype
        t_emb <- timestep_embedding(timestep, u$block_out_channels[1],
                                    flip_sin_to_cos = TRUE,
                                    downscale_freq_shift = 0L)
        t_emb <- u$time_embedding_linear_2(torch::nnf_silu(
            u$time_embedding_linear_1(t_emb$to(dtype = model_dtype))))
        ctx <- encoder_hidden_states
        if (ctx$shape[1] == d[1] && f > 1L) {
            ctx <- ctx$repeat_interleave(f, dim = 1L)
        }

        x <- sample$permute(c(1L, 3L, 2L, 4L, 5L))$reshape(c(d[1] * f, d[2], d[4], d[5]))
        x <- u$conv_in(x)
        res <- list(x)
        for (i in seq_along(u$down_blocks)) {
            blk <- u$down_blocks[[i]]
            mm <- a$down_blocks[[i]]$motion_modules
            for (j in seq_along(blk$resnets)) {
                x <- blk$resnets[[j]](x, t_emb)
                if (blk$has_attentions) {
                    x <- blk$attentions[[j]](x, ctx)
                }
                x <- mm[[j]](x, f, windows)
                res <- c(res, list(x))
                if (block_gc) gc(full = FALSE)
            }
            if (blk$has_downsamplers) {
                x <- blk$downsamplers[[1]](x)
                res <- c(res, list(x))
            }
        }

        x <- u$mid_block$resnets[[1]](x, t_emb)
        x <- u$mid_block$attentions[[1]](x, ctx)
        if (!is.null(a$mid_block)) {
            x <- a$mid_block$motion_modules[[1]](x, f, windows)
        }
        x <- u$mid_block$resnets[[2]](x, t_emb)

        for (i in seq_along(u$up_blocks)) {
            blk <- u$up_blocks[[i]]
            mm <- a$up_blocks[[i]]$motion_modules
            for (j in seq_along(blk$resnets)) {
                skip <- res[[length(res)]]
                res <- res[-length(res)]
                x <- blk$resnets[[j]](torch::torch_cat(list(x, skip), dim = 2L), t_emb)
                if (blk$has_attentions) {
                    x <- blk$attentions[[j]](x, ctx)
                }
                x <- mm[[j]](x, f, windows)
                if (block_gc) gc(full = FALSE)
            }
            if (blk$has_upsamplers) {
                x <- blk$upsamplers[[1]](x)
            }
        }

        x <- u$conv_out(torch::nnf_silu(u$conv_norm_out(x)))
        x$reshape(c(d[1], f, d[2], d[4], d[5]))$permute(c(1L, 3L, 2L, 4L, 5L))
    }
)
