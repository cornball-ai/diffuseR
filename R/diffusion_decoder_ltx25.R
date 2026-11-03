# LTX-2.5 diffusion video decoder.
#
# Ported from diffusers `models/autoencoders/ltx2_diffusion_decoder.py`
# and `pipelines/ltx2/pipeline_ltx2_diffusion_decode.py` (Apache-2.0).
# Stages 1-4 upsample the latents into a context volume with neighborhood
# attention blocks; stage 5 denoises patchified pixel noise conditioned on
# that context (one x0 step as shipped).
#
# Neighborhood attention: each query attends to a kernel-sized window
# centred on it and shifted inward at the grid border so it always holds
# exactly `kernel` positions (NATTEN's na3d; diffusers'
# _neighborhood_block_mask). R torch has neither NATTEN nor flex
# attention, so .ltx25_na3d computes it exactly with fused SDPA over
# query tiles: the windows of a tile's queries lie in one contiguous key
# region, and tiles whose queries sit at the same offsets in their region
# share one boolean mask.

# Space-to-depth on H/W: [B, C, F, H, W] -> [B, C * p^2, F, H / p, W / p],
# channels packed (channel, width offset, height offset).
.ltx25_dd_patchify <- function(x, p) {
    s <- x$shape
    x <- x$reshape(c(s[1], s[2], s[3], s[4] %/% p, p, s[5] %/% p, p))
    x <- x$permute(c(1L, 2L, 7L, 5L, 3L, 4L, 6L))
    x$reshape(c(s[1], s[2] * p * p, s[3], s[4] %/% p, s[5] %/% p))
}

.ltx25_dd_unpatchify <- function(x, p) {
    s <- x$shape
    c_out <- s[2] %/% (p * p)
    x <- x$reshape(c(s[1], c_out, p, p, s[3], s[4], s[5]))
    x <- x$permute(c(1L, 2L, 5L, 6L, 4L, 7L, 3L))
    x$reshape(c(s[1], c_out, s[3], s[4] * p, s[5] * p))
}

# Window start (0-based) of each query position along one axis.
.ltx25_na_start <- function(q, n, k) {
    pmin(pmax(q - k %/% 2L, 0L), n - k)
}

# Query tiles along one axis: list of (q = query positions, r = key region
# positions, mask = logical [length(q), length(r)], key).
.ltx25_na_axis_tiles <- function(n, k, b) {
    b <- min(b, n)
    lapply(seq(0L, n - 1L, by = b), function(a) {
        q <- a:min(a + b - 1L, n - 1L)
        ws <- .ltx25_na_start(q, n, k)
        size <- min(n, length(q) + k - 1L)
        rs <- min(ws[1], n - size)
        off <- ws - rs
        r <- rs + seq_len(size) - 1L
        mask <- outer(off, seq_len(size) - 1L,
                      function(o, j) j >= o & j < o + k)
        list(q = q, r = r, mask = mask,
             key = paste(size, paste(off, collapse = ",")))
    })
}

#' Plan a 3D neighborhood attention
#'
#' Groups query tiles of a [T, H, W] grid into SDPA calls that share one
#' mask, with token indices into the flattened (t, h, w) sequence. The
#' plan depends only on the grid and kernel, so a decoder stage builds it
#' once for all its blocks.
#'
#' @param grid Integer [T, H, W].
#' @param kernel Integer [kt, kh, kw]; each at most its grid size.
#' @param device Device for the index and mask tensors.
#' @param channels Width of the gathered keys (heads x head_dim).
#' @param tile Query tile size per axis.
#' @param budget Maximum gathered key elements (key tokens x channels)
#'   per SDPA call.
#' @return List of chunks, each list(q_idx, kv_idx, mask, n_tiles, n_q,
#'   n_k), indices 1-based long tensors.
#' @keywords internal
.ltx25_na_plan <- function(grid, kernel, device, channels = 1L,
                           tile = c(8L, 8L, 8L), budget = 2^27) {
    grid <- as.integer(grid)
    kernel <- as.integer(kernel)
    if (any(grid < kernel)) {
        stop(sprintf("Neighborhood attention needs each grid size at least its kernel size; got (T, H, W) = (%s) with kernel (%s)",
                     paste(grid, collapse = ", "), paste(kernel, collapse = ", ")))
    }
    axes <- lapply(1:3, function(a) .ltx25_na_axis_tiles(grid[a], kernel[a],
                                                          tile[a]))
    classes <- lapply(axes, function(tl) {
        keys <- vapply(tl, `[[`, "", "key")
        split(seq_along(tl), factor(keys, levels = unique(keys)))
    })
    hw <- grid[2] * grid[3]
    w_n <- grid[3]
    lin <- function(t, h, w) {
        as.vector(outer(outer(w, h * w_n, "+"), t * hw, "+")) + 1L
    }
    chunks <- list()
    for (ct in classes[[1]]) for (ch in classes[[2]]) for (cw in classes[[3]]) {
        t1 <- axes[[1]][[ct[1]]]
        h1 <- axes[[2]][[ch[1]]]
        w1 <- axes[[3]][[cw[1]]]
        n_q <- length(t1$q) * length(h1$q) * length(w1$q)
        n_k <- length(t1$r) * length(h1$r) * length(w1$r)
        mt <- torch::torch_tensor(t1$mask, device = device)
        mh <- torch::torch_tensor(h1$mask, device = device)
        mw <- torch::torch_tensor(w1$mask, device = device)
        mask <- mt$view(c(nrow(t1$mask), 1L, 1L, ncol(t1$mask), 1L, 1L)) &
            mh$view(c(1L, nrow(h1$mask), 1L, 1L, ncol(h1$mask), 1L)) &
            mw$view(c(1L, 1L, nrow(w1$mask), 1L, 1L, ncol(w1$mask)))
        mask <- mask$reshape(c(1L, 1L, n_q, n_k))
        combos <- expand.grid(w = cw, h = ch, t = ct)
        per_call <- max(1L, as.integer(budget %/% (n_k * channels)))
        for (s in seq(1L, nrow(combos), by = per_call)) {
            rows <- combos[s:min(s + per_call - 1L, nrow(combos)), ]
            q_idx <- unlist(lapply(seq_len(nrow(rows)), function(i) {
                lin(axes[[1]][[rows$t[i]]]$q, axes[[2]][[rows$h[i]]]$q,
                    axes[[3]][[rows$w[i]]]$q)
            }))
            kv_idx <- unlist(lapply(seq_len(nrow(rows)), function(i) {
                lin(axes[[1]][[rows$t[i]]]$r, axes[[2]][[rows$h[i]]]$r,
                    axes[[3]][[rows$w[i]]]$r)
            }))
            chunks[[length(chunks) + 1L]] <- list(
                q_idx = torch::torch_tensor(q_idx, dtype = torch::torch_long(),
                                            device = device),
                kv_idx = torch::torch_tensor(kv_idx, dtype = torch::torch_long(),
                                             device = device),
                mask = mask, n_tiles = nrow(rows), n_q = n_q, n_k = n_k)
        }
    }
    chunks
}

#' Exact 3D neighborhood attention
#'
#' @param query,key,value Tensors [B, T, H, W, heads, d] (query already
#'   scaled).
#' @param plan From \code{.ltx25_na_plan()} for this grid and kernel.
#' @return Tensor [B, T, H, W, heads * d].
#' @keywords internal
.ltx25_na3d <- function(query, key, value, plan) {
    s <- query$shape
    b <- s[1]
    heads <- s[5]
    d <- s[6]
    n <- s[2] * s[3] * s[4]
    flat <- function(x) x$reshape(c(b, n, heads * d))
    q <- flat(query)
    k <- flat(key)
    v <- flat(value)
    out <- torch::torch_empty(c(b, n, heads * d), dtype = query$dtype,
                              device = query$device)
    tiles <- function(x, idx, len, nt) {
        x$index_select(2L, idx)$view(c(b * nt, len, heads, d))$permute(
            c(1L, 3L, 2L, 4L))
    }
    for (ch in plan) {
        qt <- tiles(q, ch$q_idx, ch$n_q, ch$n_tiles)
        kt <- tiles(k, ch$kv_idx, ch$n_k, ch$n_tiles)
        vt <- tiles(v, ch$kv_idx, ch$n_k, ch$n_tiles)
        o <- torch::torch_scaled_dot_product_attention(qt, kt, vt,
                                                       attn_mask = ch$mask,
                                                       scale = 1)
        o <- o$permute(c(1L, 3L, 2L, 4L))$reshape(c(b, ch$n_tiles * ch$n_q,
                                                     heads * d))
        out$index_copy_(2L, ch$q_idx, o)
    }
    out$view(c(s[1], s[2], s[3], s[4], heads * d))
}

# Absolute 3D rotary embedding over (T, H, W) on x [B, T, H, W, heads,
# head_dim]: head_dim split a quarter to T, the rest halved between H and
# W, rotation pairs interleaved, computed in fp32. `t0` is the frame
# position of x's first frame (x may be a temporal slice).
.ltx25_dd_rope_split <- function(head_dim) {
    dim_t <- (head_dim %/% 4L) %/% 2L * 2L
    dim_hw <- (head_dim - dim_t) %/% 2L
    if (dim_hw %% 2L != 0L) {
        dim_t <- dim_t - 2L
        dim_hw <- (head_dim - dim_t) %/% 2L
    }
    c(dim_t, dim_hw, dim_hw)
}

.ltx25_dd_rope <- function(x, t0 = 0L, base = 10000) {
    split <- .ltx25_dd_rope_split(x$shape[6])
    f32 <- torch::torch_float32()
    out_dtype <- x$dtype
    parts <- list()
    from <- 1L
    for (axis in 1:3) {
        dim <- split[axis]
        n <- x$shape[axis + 1L]
        inv <- torch::torch_tensor(1 / base^(seq(0, dim - 1, by = 2) / dim),
                                   dtype = f32, device = x$device)
        first <- if (axis == 1L) t0 else 0L
        pos <- torch::torch_arange(first, first + n - 1L, dtype = f32,
                                   device = x$device)
        shape <- rep(1L, 6L)
        shape[axis + 1L] <- n
        shape[6] <- dim %/% 2L
        ang <- (pos$unsqueeze(2L) * inv$unsqueeze(1L))$reshape(shape)
        cos <- ang$cos()
        sin <- ang$sin()
        xs <- x$narrow(6L, from, dim)
        pairs <- xs$reshape(c(xs$shape[1:5], dim %/% 2L, 2L))
        even <- pairs[.., 1]$to(dtype = f32)
        odd <- pairs[.., 2]$to(dtype = f32)
        rot <- torch::torch_stack(list(even * cos - odd * sin,
                                       even * sin + odd * cos), dim = -1L)
        parts[[axis]] <- rot$reshape(xs$shape)$to(dtype = out_dtype)
        from <- from + dim
    }
    torch::torch_cat(parts, dim = -1L)
}

ltx25_dd_attention <- torch::nn_module(
    "ltx25_dd_attention",
    initialize = function(dim, kernel_size, head_dim = 64L) {
    self$heads <- dim %/% head_dim
    self$head_dim <- head_dim
    self$kernel_size <- as.integer(kernel_size)
    self$scale <- head_dim^-0.5
    self$to_q <- linear_noinit(dim, dim)
    self$to_k <- linear_noinit(dim, dim)
    self$to_v <- linear_noinit(dim, dim)
    self$to_out <- torch::nn_module_list(list(linear_noinit(dim, dim)))
    self$norm_q <- ltx23_rms_norm(head_dim, eps = 1e-6)
    self$norm_k <- ltx23_rms_norm(head_dim, eps = 1e-6)
},
    # Q/K/V RMS-normed, query pre-scaled, then rotated; projected a few
    # frames at a time so the fp32 rotation temporaries stay small.
    forward = function(x, plan) {
    s <- x$shape
    shape <- c(s[1:4], self$heads, self$head_dim)
    q <- torch::torch_empty(shape, dtype = x$dtype, device = x$device)
    k <- torch::torch_empty_like(q)
    v <- torch::torch_empty_like(q)
    per <- max(1L, as.integer(2^25 %/% (s[3] * s[4] * s[5])))
    for (t0 in seq(1L, s[2], by = per)) {
        len <- min(per, s[2] - t0 + 1L)
        xs <- x$narrow(2L, t0, len)
        sh <- c(s[1], len, s[3:4], self$heads, self$head_dim)
        qs <- self$norm_q(self$to_q(xs)$view(sh))$to(dtype = x$dtype)
        ks <- self$norm_k(self$to_k(xs)$view(sh))$to(dtype = x$dtype)
        q$narrow(2L, t0, len)$copy_(.ltx25_dd_rope(qs$mul(self$scale), t0 - 1L))
        k$narrow(2L, t0, len)$copy_(.ltx25_dd_rope(ks, t0 - 1L))
        v$narrow(2L, t0, len)$copy_(self$to_v(xs)$view(sh))
    }
    self$to_out[[1]](.ltx25_na3d(q, k, v, plan))
}
)

# Gated MLP w_down(silu(w_gate(x)) * w_up(x)), evaluated in tiles of 16384
# tokens to bound the hidden-width temporaries.
ltx25_dd_swiglu <- torch::nn_module(
    "ltx25_dd_swiglu",
    initialize = function(dim, hidden_dim) {
    self$w_up <- linear_noinit(dim, hidden_dim, bias = FALSE)
    self$w_gate <- linear_noinit(dim, hidden_dim, bias = FALSE)
    self$w_down <- linear_noinit(hidden_dim, dim, bias = FALSE)
},
    forward = function(x) {
    f <- function(t) self$w_down(torch::nnf_silu(self$w_gate(t)) * self$w_up(t))
    s <- x$shape
    n <- prod(s[-c(1L, length(s))])
    tile <- 16384L
    if (n <= tile) {
        return(f(x))
    }
    flat <- x$reshape(c(s[1], n, s[length(s)]))
    out <- torch::torch_empty_like(flat)
    for (a in seq(1L, n, by = tile)) {
        len <- min(tile, n - a + 1L)
        out$narrow(2L, a, len)$copy_(f(flat$narrow(2L, a, len)))
    }
    out$reshape(s)
}
)

.ltx25_dd_hidden <- function(dim, mlp_ratio = 4) {
    (as.integer(dim * mlp_ratio) + 15L) %/% 16L * 16L
}

ltx25_dd_na_block <- torch::nn_module(
    "ltx25_dd_na_block",
    initialize = function(dim, kernel_size, head_dim = 64L) {
    self$norm1 <- ltx23_rms_norm(dim, eps = 1e-6)
    self$attn <- ltx25_dd_attention(dim, kernel_size, head_dim)
    self$norm2 <- ltx23_rms_norm(dim, eps = 1e-6)
    self$mlp <- ltx25_dd_swiglu(dim, .ltx25_dd_hidden(dim))
},
    forward = function(x, plan) {
    x <- x + self$attn(self$norm1(x), plan)
    x + self$mlp(self$norm2(x))
}
)

ltx25_dd_diffusion_block <- torch::nn_module(
    "ltx25_dd_diffusion_block",
    initialize = function(dim, kernel_size, context_channels, head_dim = 64L,
                          num_mod_params = 7L) {
    self$context_proj <- linear_noinit(context_channels, dim)
    self$scale_shift_table <- torch::nn_parameter(torch::torch_zeros(num_mod_params, dim))
    self$norm1 <- ltx23_rms_norm(dim, eps = 1e-6)
    self$attn <- ltx25_dd_attention(dim, kernel_size, head_dim)
    self$norm2 <- ltx23_rms_norm(dim, eps = 1e-6)
    self$mlp <- ltx25_dd_swiglu(dim, .ltx25_dd_hidden(dim))
},
    forward = function(x, context, modulation, plan) {
    mod <- function(i) {
        modulation[[i]] + self$scale_shift_table[i, ]$view(c(1L, 1L, 1L, 1L, -1L))
    }
    x <- x + self$context_proj(context)
    x <- x + self$attn(self$norm1(x) * mod(1)$add(1) + mod(2), plan)
    x + self$mlp(self$norm2(x) * mod(4)$add(1) + mod(5))
}
)

# Shared AdaLN-Zero: timestep embedding -> 7 modulation chunks (only the
# four scale/shift chunks are used; the residuals are ungated).
ltx25_dd_adaln <- torch::nn_module(
    "ltx25_dd_adaln",
    initialize = function(t_emb_dim, dim) {
    self$proj <- linear_noinit(t_emb_dim, 7L * dim)
}
)

# Linear channel expansion then a channels-last pixel shuffle; a temporal
# stride of 2 makes a duplicate leading frame, dropped to keep the causal
# 1 + 8k frame mapping.
ltx25_dd_upsampler <- torch::nn_module(
    "ltx25_dd_upsampler",
    initialize = function(in_channels, stride, reduction = 1L) {
    self$stride <- as.integer(stride)
    proj_out <- prod(self$stride) * in_channels %/% reduction
    self$out_channels <- proj_out %/% prod(self$stride)
    self$proj <- linear_noinit(in_channels, proj_out)
},
    forward = function(x, drop_leading_frame = TRUE) {
    s <- x$shape
    st <- self$stride
    x <- self$proj(x)$reshape(c(s[1:4], self$out_channels, st))
    x <- x$permute(c(1L, 2L, 6L, 3L, 7L, 4L, 8L, 5L))
    x <- x$reshape(c(s[1], s[2] * st[1], s[3] * st[2], s[4] * st[3],
                     self$out_channels))
    if (st[1] == 2L && drop_leading_frame) {
        x <- x$narrow(2L, 2L, x$shape[2] - 1L)
    }
    x
}
)

#' LTX-2.5 diffusion video decoder module
#'
#' The decoder half of LTX-2.5's default video VAE: a small diffusion
#' model that renders pixels from the same latents the convolutional
#' decoder reads. Built by \code{\link{ltx25_load_diffusion_decoder}};
#' arguments default to the shipped configuration.
#'
#' @param latent_channels,out_channels,patch_size,head_dim,stage_channels,stage_depths,stage_kernels,upsample_strides,upsample_reductions,stage5_kernel,t_emb_dim
#'   Architecture (diffusers \code{LTX2VideoDiffusionDecoder3d}).
#' @return An \code{nn_module}; \code{$decode(z)} takes denormalized
#'   latents [B, 128, F, H/32, W/32] and returns pixels [B, 3, 8(F-1)+1,
#'   H, W] in about [-1, 1].
#' @export
ltx25_diffusion_decoder <- torch::nn_module(
    "ltx25_diffusion_decoder",
    initialize = function(latent_channels = 128L, out_channels = 3L,
                          patch_size = 4L, head_dim = 64L,
                          stage_channels = c(2048L, 1024L, 512L, 512L, 256L),
                          stage_depths = c(4L, 6L, 4L, 2L, 8L),
                          stage_kernels = list(c(3L, 7L, 7L), c(3L, 7L, 7L),
                                               c(3L, 5L, 5L), c(3L, 5L, 5L)),
                          upsample_strides = list(c(1L, 2L, 2L), c(2L, 1L, 1L),
                                                  c(2L, 2L, 2L), c(2L, 2L, 2L)),
                          upsample_reductions = c(2L, 2L, 1L, 2L),
                          stage5_kernel = c(11L, 11L, 11L), t_emb_dim = 384L) {
    self$patch_size <- patch_size
    self$out_channels <- out_channels
    self$stage_kernels <- stage_kernels
    self$stage5_kernel <- as.integer(stage5_kernel)
    self$temporal_ratio <- prod(vapply(upsample_strides, `[`, 1L, 1L))
    self$trailing_pad <- (stage_kernels[[1]][1] %/% 2L) * 2L
    self$conv_in <- linear_noinit(latent_channels, stage_channels[1])
    self$det_stages <- torch::nn_module_list(lapply(seq_along(upsample_strides), function(i) {
        torch::nn_module_list(lapply(seq_len(stage_depths[i]), function(j) {
            ltx25_dd_na_block(stage_channels[i], stage_kernels[[i]], head_dim)
        }))
    }))
    self$upsamples <- torch::nn_module_list(lapply(seq_along(upsample_strides), function(i) {
        ltx25_dd_upsampler(stage_channels[i], upsample_strides[[i]],
                           upsample_reductions[i])
    }))
    self$t_embedder <- ltx23_combined_timestep_embed(t_emb_dim)
    c5 <- stage_channels[length(stage_channels)]
    pix <- out_channels * patch_size^2
    self$conv_in_x_t <- linear_noinit(pix, c5)
    self$shared_adaln <- ltx25_dd_adaln(t_emb_dim, c5)
    self$diff_blocks <- torch::nn_module_list(lapply(seq_len(stage_depths[length(stage_depths)]), function(j) {
        ltx25_dd_diffusion_block(c5, stage5_kernel, c5, head_dim)
    }))
    self$norm_out <- ltx23_rms_norm(c5, eps = 1e-6)
    self$conv_out <- linear_noinit(c5, pix)
    self$latents_mean <- torch::nn_buffer(torch::torch_zeros(latent_channels))
    self$latents_std <- torch::nn_buffer(torch::torch_ones(latent_channels))
    self$tiling <- NULL
},
    run_stage = function(i, x) {
    plan <- .ltx25_na_plan(x$shape[2:4], self$stage_kernels[[i]], x$device,
                           channels = x$shape[5])
    blocks <- self$det_stages[[i]]
    for (j in seq_len(length(blocks))) {
        blk <- blocks[[j]]
        x <- blk(x, plan)
    }
    x
},
    # Latents [B, C, T, H, W] -> channels-last features entering the last
    # deterministic stage. The last latent frame is repeated
    # `trailing_pad` times (ghost frames, moving the inward-shifted border
    # windows past the kept frames); stage4() crops them.
    stages_1_to_3 = function(z) {
    pad <- self$trailing_pad
    if (pad > 0L) {
        last <- z$narrow(3L, z$shape[3], 1L)$expand(c(-1L, -1L, pad, -1L, -1L))
        z <- torch::torch_cat(list(z, last), dim = 3L)
    }
    x <- self$conv_in(z$permute(c(1L, 3L, 4L, 5L, 2L)))
    n <- length(self$upsamples)
    for (i in seq_len(n - 1L)) {
        x <- self$upsamples[[i]](self$run_stage(i, x))
    }
    x
},
    # Last deterministic stage -> context [B, F, H/8, W/8, C5]. A temporal
    # tile not containing t = 0 keeps the upsample's duplicate leading
    # frame; only the tile reaching the video end crops the ghost frames.
    stage4 = function(x, drop_leading_frame = TRUE, crop_trailing_ghost = TRUE) {
    n <- length(self$upsamples)
    x <- self$upsamples[[n]](self$run_stage(n, x),
                             drop_leading_frame = drop_leading_frame)
    pad <- self$trailing_pad
    if (crop_trailing_ghost && pad > 0L) {
        x <- x$narrow(2L, 1L, x$shape[2] - pad * self$temporal_ratio)
    }
    x
},
    context = function(z) {
    self$stage4(self$stages_1_to_3(z))
},
    # One stage-5 step: the model's pixel prediction [B, 3, F, H, W].
    step = function(context, x_t, timestep) {
    t_emb <- self$t_embedder(timestep * 1000, hidden_dtype = context$dtype)
    chunks <- self$shared_adaln$proj(torch::nnf_silu(t_emb))$chunk(7L, dim = -1L)
    modulation <- lapply(chunks, function(m) m$view(c(m$shape[1], 1L, 1L, 1L, -1L)))
    x <- self$conv_in_x_t(.ltx25_dd_patchify(x_t, self$patch_size)$permute(
        c(1L, 3L, 4L, 5L, 2L)))
    plan <- .ltx25_na_plan(x$shape[2:4], self$stage5_kernel, x$device,
                           channels = x$shape[5])
    for (j in seq_len(length(self$diff_blocks))) {
        blk <- self$diff_blocks[[j]]
        x <- blk(x, context, modulation, plan)
    }
    x <- self$conv_out(self$norm_out(x)$to(dtype = context$dtype))
    .ltx25_dd_unpatchify(x$permute(c(1L, 5L, 2L, 3L, 4L))$contiguous(),
                         self$patch_size)
},
    # Denoise x_t from t = 1 over `steps` x0-prediction steps.
    denoise = function(context, x_t, steps = 1L) {
    b <- context$shape[1]
    ts <- seq(1, 1 / steps, length.out = steps)
    tt <- function(v) torch::torch_full(b, v, dtype = torch::torch_float32(),
                                        device = context$device)
    if (steps == 1L) {
        return(self$step(context, x_t, tt(1)))
    }
    for (i in seq_len(steps)) {
        t_next <- if (i < steps) ts[i + 1L] else 0
        x0 <- self$step(context, x_t, tt(ts[i]))$to(dtype = torch::torch_float32())
        xf <- x_t$to(dtype = torch::torch_float32())
        vel <- (xf - x0) / ts[i]
        x_t <- (xf - vel * (ts[i] - t_next))$to(dtype = x_t$dtype)
    }
    x_t
},
    # Tiled decoding: stage 4 and stage 5 (which dominate memory) run on
    # overlapping tiles blended linearly; stages 1-3 see the whole latent.
    # Sizes are in decoded pixels / frames: tile and the distance between
    # tile starts; their difference is the blended overlap. Multiples of
    # 8 px and 2 frames.
    enable_tiling = function(height = 512L, width = 512L, frames = 64L,
                             stride_height = 448L, stride_width = 448L,
                             stride_frames = 40L) {
    self$tiling <- list(height = height, width = width, frames = frames,
                        stride_height = stride_height,
                        stride_width = stride_width,
                        stride_frames = stride_frames)
    invisible(self)
},
    disable_tiling = function() {
    self$tiling <- NULL
    invisible(self)
},
    # z: denormalized latents. noise: optional pixel noise for the whole
    # canvas (else drawn; a single-step tiled decode draws per tile).
    decode = function(z, steps = 1L, noise = NULL) {
    tl <- self$tiling
    if (!is.null(tl) &&
        (z$shape[3] > tl$frames %/% 8L || z$shape[4] > tl$height %/% 32L ||
         z$shape[5] > tl$width %/% 32L)) {
        return(self$tiled_decode(z, steps, noise))
    }
    context <- self$context(z)
    p <- self$patch_size
    shape <- c(z$shape[1], self$out_channels, context$shape[2],
               context$shape[3] * p, context$shape[4] * p)
    if (is.null(noise)) {
        noise <- torch::torch_randn(shape, dtype = z$dtype, device = z$device)
    }
    self$denoise(context, noise$to(dtype = z$dtype, device = z$device), steps)
},
    tiled_decode = function(z, steps = 1L, noise = NULL) {
    tl <- self$tiling
    p <- self$patch_size
    n_up <- length(self$upsamples)
    up <- self$upsamples[[n_up]]$stride
    scale <- c(up[1], up[2] * p, up[3] * p)
    tile <- c(tl$frames, tl$height, tl$width) %/% scale
    stride <- c(tl$stride_frames, tl$stride_height, tl$stride_width) %/% scale
    # each tile must hold the stage-4 kernel and, scaled by the upsample,
    # the stage-5 kernel
    min_size <- pmax(self$stage_kernels[[n_up]], -(-self$stage5_kernel %/% up))
    features <- self$stages_1_to_3(z)
    ghost <- self$trailing_pad * prod(vapply(seq_len(n_up - 1L), function(i) {
        self$upsamples[[i]]$stride[1]
    }, 1L))
    n_t <- features$shape[2] - ghost
    n_h <- features$shape[3]
    n_w <- features$shape[4]
    t_tiles <- .ltx25_tile_intervals(n_t, tile[1], stride[1], min_size[1])
    h_tiles <- .ltx25_tile_intervals(n_h, tile[2], stride[2], min_size[2])
    w_tiles <- .ltx25_tile_intervals(n_w, tile[3], stride[3], min_size[3])
    blend <- (tile - stride) * scale
    single <- steps == 1L && is.null(noise)
    noise_full <- noise
    if (!single && is.null(noise_full)) {
        frames <- n_t * scale[1] - (if (scale[1] == 2L) 1L else 0L)
        noise_full <- torch::torch_randn(
            c(z$shape[1], self$out_channels, frames, n_h * scale[2],
              n_w * scale[3]), dtype = z$dtype, device = z$device)
    }
    groups <- list()
    for (ti in seq_along(t_tiles)) {
        t0 <- t_tiles[[ti]][1]
        t1 <- t_tiles[[ti]][2]
        origin <- t0 == 0L
        trailing <- t1 == n_t
        f_t1 <- if (trailing) features$shape[2] else t1
        rows <- list()
        for (hi in seq_along(h_tiles)) {
            row <- list()
            for (wi in seq_along(w_tiles)) {
                h <- h_tiles[[hi]]
                w <- w_tiles[[wi]]
                ctx <- self$stage4(
                    features$narrow(2L, t0 + 1L, f_t1 - t0)$narrow(
                        3L, h[1] + 1L, h[2] - h[1])$narrow(4L, w[1] + 1L, w[2] - w[1]),
                    drop_leading_frame = origin, crop_trailing_ghost = trailing)
                shape <- c(z$shape[1], self$out_channels, ctx$shape[2],
                           ctx$shape[3] * p, ctx$shape[4] * p)
                x_t <- if (single) {
                    torch::torch_randn(shape, dtype = z$dtype, device = z$device)
                } else {
                    # a non-origin tile keeps the duplicate leading frame,
                    # so its first cell starts one pixel frame earlier
                    pt0 <- t0 * scale[1] -
                        (if (!origin && scale[1] == 2L) 1L else 0L)
                    noise_full$narrow(3L, pt0 + 1L, shape[3])$narrow(
                        4L, h[1] * scale[2] + 1L, shape[4])$narrow(
                        5L, w[1] * scale[3] + 1L, shape[5])
                }
                row[[wi]] <- self$denoise(ctx, x_t, steps)
                rm(ctx, x_t)
            }
            rows[[hi]] <- row
        }
        result_rows <- list()
        for (i in seq_along(rows)) {
            result_row <- list()
            for (j in seq_along(rows[[i]])) {
                tile_ij <- rows[[i]][[j]]
                if (i > 1L) {
                    tile_ij <- .ltx23_blend_v(rows[[i - 1L]][[j]], tile_ij, blend[2])
                }
                if (j > 1L) {
                    tile_ij <- .ltx23_blend_h(rows[[i]][[j - 1L]], tile_ij, blend[3])
                }
                rows[[i]][[j]] <- tile_ij
                keep_h <- if (i < length(rows)) stride[2] * scale[2] else tile_ij$shape[4]
                keep_w <- if (j < length(rows[[i]])) stride[3] * scale[3] else tile_ij$shape[5]
                result_row[[j]] <- tile_ij$narrow(4L, 1L, keep_h)$narrow(5L, 1L, keep_w)
            }
            result_rows[[i]] <- torch::torch_cat(result_row, dim = 5L)
        }
        groups[[ti]] <- torch::torch_cat(result_rows, dim = 4L)
        rm(rows, result_rows)
    }
    out <- list()
    for (k in seq_along(groups)) {
        g <- groups[[k]]
        if (k > 1L) {
            g <- .ltx23_blend_t(groups[[k - 1L]], g, blend[1])
            groups[[k]] <- g
        }
        if (k < length(groups)) {
            # the origin group is one frame short under the causal mapping
            keep <- stride[1] * scale[1] -
                (if (k == 1L && scale[1] == 2L) 1L else 0L)
            g <- g$narrow(3L, 1L, keep)
        }
        out[[k]] <- g
    }
    torch::torch_cat(out, dim = 3L)
}
)

# Overlapping [start, end) tiles (0-based) covering [0, n) with starts
# `stride` apart; a trailing remnant shorter than `min_size` merges into
# the previous tile (neighborhood attention rejects grids under its
# kernel).
.ltx25_tile_intervals <- function(n, tile, stride, min_size) {
    if (n <= tile) {
        return(list(c(0L, n)))
    }
    starts <- seq(0L, n - 1L, by = stride)
    while (length(starts) > 1L && n - starts[length(starts)] < min_size) {
        starts <- starts[-length(starts)]
    }
    k <- length(starts)
    c(lapply(starts[-k], function(s) c(s, min(s + tile, n))),
      list(c(starts[k], n)))
}

#' Map an LTX-2.5 diffusion decoder checkpoint key
#'
#' @param key Checkpoint key (\code{decoder.*} or
#'   \code{per_channel_statistics.*}, optionally under \code{vae.}).
#' @return The module key, or \code{NA} for keys the decoder does not use
#'   (the encoder, and \code{type_emb}, which tags keyframe latents in
#'   Lightricks' keyframe-aware decode).
#' @keywords internal
.ltx25_dd_map_key <- function(key) {
    key <- sub("^vae\\.", "", key)
    if (key == "per_channel_statistics.mean-of-means") return("latents_mean")
    if (key == "per_channel_statistics.std-of-means") return("latents_std")
    if (!startsWith(key, "decoder.") || key == "decoder.type_emb" ||
        grepl("coarse_", key, fixed = TRUE)) {
        return(NA_character_)
    }
    key <- sub("^decoder\\.", "", key)
    key <- sub("t_embedder.mlp.0.", "t_embedder.timestep_embedder.linear_1.", key, fixed = TRUE)
    key <- sub("t_embedder.mlp.2.", "t_embedder.timestep_embedder.linear_2.", key, fixed = TRUE)
    key <- sub(".attn.proj.", ".attn.to_out.0.", key, fixed = TRUE)
    key <- sub(".attn.q_norm.", ".attn.norm_q.", key, fixed = TRUE)
    sub(".attn.k_norm.", ".attn.norm_k.", key, fixed = TRUE)
}

#' Locate (or download) the LTX-2.5 diffusion decoder file
#'
#' The decoder ships as LTX-2.5's default video VAE,
#' \code{vae/ltx-2.5-video-vae-bf16.safetensors} (1.5 GB) in the gated
#' \code{Lightricks/LTX-2.5} repository: accept the LTX-2 Community
#' License on Hugging Face and set \code{HF_TOKEN}. The weights are read
#' from the Hugging Face cache and never redistributed.
#'
#' @param download Logical. Download the file when it is not cached
#'   (asks first in interactive sessions; set
#'   \code{options(diffuseR.consent = TRUE)} otherwise).
#' @return The local file path.
#' @export
ltx25_diffusion_decoder_path <- function(download = FALSE) {
    if (!requireNamespace("hfhub", quietly = TRUE)) {
        stop("The hfhub package is required to locate model weights.")
    }
    repo <- "Lightricks/LTX-2.5"
    file <- "vae/ltx-2.5-video-vae-bf16.safetensors"
    path <- tryCatch(hfhub::hub_download(repo, file, local_files_only = TRUE),
                     error = function(e) NULL)
    if (is.null(path) && isTRUE(download) &&
        .ltx23_consent("the LTX-2.5 diffusion video decoder (1.5 GB)")) {
        path <- hfhub::hub_download(repo, file)
    }
    if (is.null(path)) {
        stop("The LTX-2.5 diffusion decoder is not cached; call ",
             "ltx25_diffusion_decoder_path(download = TRUE) (gated: accept ",
             "the license for ", repo, " and set HF_TOKEN)", call. = FALSE)
    }
    path
}

#' Load the LTX-2.5 diffusion video decoder
#'
#' @param path The 2.5 video VAE file
#'   (\code{vae/ltx-2.5-video-vae-bf16.safetensors}, the diffusion
#'   decoder; the encoder half is ignored).
#' @param device,dtype Where and in what precision to build it.
#' @param ... Architecture arguments for \code{\link{ltx25_diffusion_decoder}}
#'   (the defaults are the shipped decoder).
#' @return An \code{\link{ltx25_diffusion_decoder}} in eval mode.
#' @export
ltx25_load_diffusion_decoder <- function(path, device = "cpu",
                                         dtype = torch::torch_bfloat16(), ...) {
    ckpt <- ltx23_open_checkpoint(path, require_version = "2.5")
    keys <- ckpt$keys
    if (!any(grepl("^(vae\\.)?decoder\\.diff_blocks\\.", keys))) {
        stop("Not a diffusion video decoder file: ", path)
    }
    if (any(grepl("\\.gate_(msa|mlp|ctx)$", keys))) {
        stop("Gated diffusion decoder checkpoints are not supported: ", path)
    }
    model <- .construct_skeleton(ltx25_diffusion_decoder, ..., dtype = dtype)
    params <- c(model$named_parameters(), model$named_buffers())
    seen <- character()
    torch::with_no_grad({
        for (key in keys) {
            name <- .ltx25_dd_map_key(key)
            if (is.na(name)) next
            val <- ckpt$handle$get_tensor(key)
            if (grepl("\\.attn\\.qkv\\.(weight|bias)$", name)) {
                leaf <- sub(".*\\.", "", name)
                prefix <- sub("qkv\\.(weight|bias)$", "", name)
                chunk <- val$shape[1] %/% 3L
                parts <- list(to_q = 1L, to_k = chunk + 1L, to_v = 2L * chunk + 1L)
                for (p in names(parts)) {
                    nm <- paste0(prefix, p, ".", leaf)
                    .ltx25_dd_assign(params, nm, val$narrow(1L, parts[[p]], chunk))
                    seen <- c(seen, nm)
                }
                next
            }
            .ltx25_dd_assign(params, name, val)
            seen <- c(seen, name)
        }
    })
    missing <- setdiff(names(params), seen)
    if (length(missing)) {
        stop("Diffusion decoder weights missing from ", path, ": ",
             paste(utils::head(missing, 5L), collapse = ", "))
    }
    model$to(device = device)
    model$eval()
    model
}

.ltx25_dd_assign <- function(params, name, val) {
    p <- params[[name]]
    if (is.null(p)) {
        stop("Unexpected diffusion decoder key: ", name)
    }
    if (!identical(as.integer(p$shape), as.integer(val$shape))) {
        stop(sprintf("Shape mismatch for %s: [%s] vs [%s]", name,
                     paste(p$shape, collapse = ", "),
                     paste(val$shape, collapse = ", ")))
    }
    p$copy_(val$to(dtype = p$dtype))
}
