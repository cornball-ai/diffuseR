#' Stable Diffusion XL UNet (anvl port of unet_sdxl_native)
#'
#' anvl re-implementation of \code{diffuseR::unet_sdxl_native}: the SDXL
#' UNet2DConditionModel. \code{conv_in}; three down blocks (two ResNet
#' blocks each; the first is plain ResNet + stride-2 downsample, the
#' second and third add a SpatialTransformer of depth 2 and 10 after
#' every ResNet, and the third has no downsample); the mid block (ResNet,
#' depth-10 SpatialTransformer, ResNet); three up blocks (three ResNet
#' blocks each, concatenating the LIFO down-path skips before every
#' ResNet, cross-attention on the first two blocks with depth 10 and 2,
#' and a nearest-2x upsample conv on all but the last); the sinusoidal
#' timestep embedding + MLP; the SDXL added-conditioning path (pooled
#' text embeds concatenated with the sinusoidal time-ids, projected by a
#' two-layer MLP and added to the timestep embedding); and the
#' GroupNorm-32 -> SiLU -> conv output head. The SpatialTransformer is
#' GroupNorm -> linear proj_in -> N BasicTransformerBlocks
#' (LayerNorm->self-attn, LayerNorm->cross-attn to the text embeddings,
#' LayerNorm->GEGLU feed-forward, each residual) -> linear proj_out +
#' residual.
#'
#' \code{cross_attention_dim} is the concatenated CLIP-L + OpenCLIP
#' bigG text width (2048); \code{attention_head_dim} is 64; the
#' transformer depth is 0/2/10 down, 10 mid, 10/2/0 up. SDXL uses
#' linear (not 1x1-conv) proj_in / proj_out.
#'
#' The conv helpers (\code{yq_conv3x3}, \code{yq_conv1x1},
#' \code{yq_add_conv_bias}) are shared with the VAE port
#' (\code{R/anvl_vae.R}).
#'
#' @name anvl_sdxl_unet
NULL

# GroupNorm and LayerNorm epsilon (torch nn_group_norm / nn_layer_norm
# both default to 1e-5, matching group_norm_32 and nn_layer_norm here).
.YQ_SDXL_EPS <- 1e-5

# Exact GELU (erf form, matching torch nnf_gelu default approximate="none").
.yq_sdxl_gelu <- function(x) {
    0.5 * x * (anvl::nv_erf(x * (1 / sqrt(2))) + 1)
}

# Affine LayerNorm over the last dim: yq_layer_norm (no affine) * w + b,
# with w/b [D] broadcast over [..., D].
.yq_sdxl_ln <- function(x, weight, bias, eps = .YQ_SDXL_EPS) {
    s <- anvl::shape(x)
    yunque::yq_layer_norm(x, eps = eps) *
    anvl::nv_broadcast_to(weight, s) + anvl::nv_broadcast_to(bias, s)
}

# Multi-head attention (self or cross). query from `x` [B, Sq, inner];
# key/value from `context` [B, Sk, *]. Weights: to_q_w/to_k_w/to_v_w
# (bias-free [in, inner]), to_out_w/to_out_b. Scale 1/sqrt(head_dim) via
# yq_sdpa.
.yq_sdxl_attn <- function(x, context, w, n_heads, head_dim) {
    sx <- anvl::shape(x)
    b <- sx[1L]
    sq <- sx[2L]
    sk <- anvl::shape(context)[2L]
    inner <- n_heads * head_dim
    q <- yunque::yq_linear(x, w$to_q_w)
    k <- yunque::yq_linear(context, w$to_k_w)
    v <- yunque::yq_linear(context, w$to_v_w)
    # [B, S, inner] -> [B, S, H, D] -> [B, H, S, D]
    to_heads <- function(t, sl) anvl::nv_transpose(
        anvl::nv_reshape(t, c(b, sl, n_heads, head_dim)), c(1L, 3L, 2L, 4L))
    q <- to_heads(q, sq)
    k <- to_heads(k, sk)
    v <- to_heads(v, sk)
    attn <- yunque::yq_sdpa(q, k, v)                 # [B, H, Sq, D]
    out <- anvl::nv_reshape(anvl::nv_transpose(attn, c(1L, 3L, 2L, 4L)),
                            c(b, sq, inner))
    yunque::yq_linear(out, w$to_out_w, w$to_out_b)
}

# GEGLU feed-forward: linear -> [.., 2*inner]; gate = second half; return
# linear(first * gelu(gate)).
.yq_sdxl_ff <- function(x, w) {
    proj <- yunque::yq_linear(x, w$ff_in_w, w$ff_in_b)
    half <- anvl::shape(proj)[anvl::ndims(proj)] %/% 2L
    a <- yunque::yq_slice_lastdim(proj, 1L, half)
    g <- yunque::yq_slice_lastdim(proj, half + 1L, 2L * half)
    yunque::yq_linear(a * .yq_sdxl_gelu(g), w$ff_out_w, w$ff_out_b)
}

# BasicTransformerBlock: residual self-attn, residual cross-attn to
# `context`, residual GEGLU feed-forward, each preceded by an affine
# LayerNorm.
.yq_sdxl_transformer_block <- function(x, context, w, n_heads, head_dim) {
    n1 <- .yq_sdxl_ln(x, w$norm1_w, w$norm1_b)
    x <- x + .yq_sdxl_attn(n1, n1, w$attn1, n_heads, head_dim)
    n2 <- .yq_sdxl_ln(x, w$norm2_w, w$norm2_b)
    x <- x + .yq_sdxl_attn(n2, context, w$attn2, n_heads, head_dim)
    n3 <- .yq_sdxl_ln(x, w$norm3_w, w$norm3_b)
    x + .yq_sdxl_ff(n3, w$ff)
}

# SpatialTransformer: GroupNorm -> [B,C,H,W] to [B,H*W,C] -> linear
# proj_in -> N transformer blocks -> linear proj_out -> back to
# [B,C,H,W] + residual.
.yq_sdxl_spatial_transformer <- function(x, context, w, n_heads, head_dim) {
    s <- anvl::shape(x)
    b <- s[1L]; c <- s[2L]; h <- s[3L]; wd <- s[4L]
    x_in <- x
    xn <- yunque::yq_group_norm(x, w$gn_w, w$gn_b, 32L, .YQ_SDXL_EPS)
    seq <- anvl::nv_reshape(anvl::nv_transpose(xn, c(1L, 3L, 4L, 2L)),
                            c(b, h * wd, c))
    seq <- yunque::yq_linear(seq, w$proj_in_w, w$proj_in_b)
    for (tb in w$blocks) {
        seq <- .yq_sdxl_transformer_block(seq, context, tb, n_heads, head_dim)
    }
    seq <- yunque::yq_linear(seq, w$proj_out_w, w$proj_out_b)
    out <- anvl::nv_transpose(anvl::nv_reshape(seq, c(b, h, wd, c)),
                              c(1L, 4L, 2L, 3L))
    out + x_in
}

# ResNet block: GroupNorm -> SiLU -> conv3x3, add SiLU(emb) projected to
# [B, out] broadcast over spatial, GroupNorm -> SiLU -> conv3x3, plus skip
# (conv1x1 shortcut when channels change).
.yq_sdxl_resnet <- function(x, emb, w) {
    h <- yunque::yq_group_norm(x, w$norm1_w, w$norm1_b, 32L, .YQ_SDXL_EPS)
    h <- yq_conv3x3(yunque::yq_silu(h), w$conv1_w, w$conv1_b)
    t <- yunque::yq_linear(yunque::yq_silu(emb), w$time_emb_w, w$time_emb_b)
    ts <- anvl::shape(t)
    hs <- anvl::shape(h)
    t <- anvl::nv_reshape(t, c(ts[1L], ts[2L], 1L, 1L))
    h <- h + anvl::nv_broadcast_to(t, hs)
    h <- yunque::yq_group_norm(h, w$norm2_w, w$norm2_b, 32L, .YQ_SDXL_EPS)
    h <- yq_conv3x3(yunque::yq_silu(h), w$conv2_w, w$conv2_b)
    if (!is.null(w$shortcut_w)) {
        x <- yq_conv1x1(x, w$shortcut_w, w$shortcut_b)
    }
    h + x
}

# Downsample: 3x3 stride-2 pad-1 conv (halves H/W).
.yq_sdxl_downsample <- function(x, w) {
    y <- anvl::nv_conv2d(x, w$conv_w, stride = 2L, padding = 1L)
    yq_add_conv_bias(y, w$conv_b)
}

# Upsample: nearest-2x then 3x3 pad-1 conv.
.yq_sdxl_upsample <- function(x, w) {
    yq_conv3x3(yunque::yq_upsample_nearest2d(x), w$conv_w, w$conv_b)
}

# Static per-block configuration derived from the channel plan and the
# transformer-depth plan. down blocks: downsample on all but the last,
# attention where depth > 0; up blocks (reversed channels/depths):
# upsample on all but the last, attention where depth > 0, one extra
# ResNet for the skip.
.yq_sdxl_config <- function(block_out_channels, layers_per_block,
                            transformer_layers_per_block, attention_head_dim) {
    nblocks <- length(block_out_channels)
    head_dim <- as.integer(attention_head_dim)
    tdepth <- as.integer(transformer_layers_per_block)
    down <- lapply(seq_len(nblocks), function(i) {
        is_final <- i == nblocks
        ch <- block_out_channels[i]
        list(out = ch, n_heads = ch %/% head_dim, depth = tdepth[i],
             has_attn = tdepth[i] > 0L, has_down = !is_final,
             nres = as.integer(layers_per_block))
    })
    reversed <- rev(block_out_channels)
    rdepth <- rev(tdepth)
    up <- lapply(seq_len(nblocks), function(i) {
        ch <- reversed[i]
        list(out = ch, n_heads = ch %/% head_dim, depth = rdepth[i],
             has_attn = rdepth[i] > 0L, has_up = i != nblocks,
             nres = as.integer(layers_per_block) + 1L)
    })
    list(down = down, up = up,
         mid_heads = block_out_channels[nblocks] %/% head_dim,
         mid_depth = tdepth[nblocks], head_dim = head_dim, nblocks = nblocks)
}

#' Host-side sinusoidal timestep embedding for the SDXL UNet
#'
#' Precomputes the parameter-free sinusoidal embedding (the input to the
#' UNet's time-embedding MLP), mirroring \code{diffuseR}'s
#' \code{timestep_embedding(timestep, dim, flip_sin_to_cos = TRUE,
#' downscale_freq_shift = 0)}: \code{[cos(args), sin(args)]} with
#' \code{args = t * exp(-log(10000) / half_dim * (0:(half_dim-1)))}.
#' Computed in base R and returned as an \code{AnvlArray} so the jit
#' boundary stays weight-only.
#'
#' @param timestep Numeric vector \code{[batch]} of timesteps.
#' @param dim Integer. Embedding width (SDXL main embed: 320, even).
#' @param device Character. Target device.
#'
#' @return AnvlArray \code{[batch, dim]}, f32.
#'
#' @export
yq_sdxl_time_embed <- function(timestep, dim = 320L, device = "cpu") {
    dim <- as.integer(dim)
    half_dim <- dim %/% 2L
    emb_scale <- log(10000) / half_dim
    freqs <- exp(-emb_scale * (0:(half_dim - 1L)))
    args <- outer(as.numeric(timestep), freqs)          # [batch, half_dim]
    emb <- cbind(cos(args), sin(args))                  # [batch, dim]
    anvl::nv_array(emb, dtype = "f32", device = device)
}

#' Host-side SDXL added-conditioning time-ids embedding
#'
#' Reproduces the SDXL \code{add_embedding} Fourier features:
#' \code{timestep_embedding(time_ids$flatten(), addition_time_embed_dim)}
#' reshaped to \code{[batch, n_time_ids * dim]}. Each of the
#' \code{n_time_ids} (6) micro-conditioning scalars per batch row is
#' sinusoidally embedded (same \code{[cos, sin]} convention as
#' \code{\link{yq_sdxl_time_embed}}) and the row-major concatenation
#' becomes the \code{time_ids} half of the \code{add_embedding} input
#' (the pooled text embeds are the other half).
#'
#' @param time_ids Numeric matrix \code{[batch, n_time_ids]} (a length-
#'   \code{n_time_ids} vector is treated as one batch row).
#' @param dim Integer. Per-id embedding width (SDXL: 256, even).
#' @param device Character. Target device.
#'
#' @return AnvlArray \code{[batch, n_time_ids * dim]}, f32.
#'
#' @export
yq_sdxl_time_ids_embed <- function(time_ids, dim = 256L, device = "cpu") {
    dim <- as.integer(dim)
    if (is.null(dim(time_ids))) time_ids <- matrix(time_ids, nrow = 1L)
    nb <- nrow(time_ids)
    ntid <- ncol(time_ids)
    half_dim <- dim %/% 2L
    emb_scale <- log(10000) / half_dim
    freqs <- exp(-emb_scale * (0:(half_dim - 1L)))
    flat <- as.numeric(t(time_ids))                 # row-major flatten [nb*ntid]
    args <- outer(flat, freqs)                      # [nb*ntid, half_dim]
    emb <- cbind(cos(args), sin(args))              # [nb*ntid, dim] in flat order
    # torch reshape [nb*ntid, dim] -> [nb, ntid*dim] (row-major): each row
    # is the row-major concatenation of its ntid embeddings.
    out <- matrix(0, nrow = nb, ncol = ntid * dim)
    for (bi in seq_len(nb)) {
        rows <- ((bi - 1L) * ntid + 1L):(bi * ntid)
        out[bi, ] <- as.numeric(t(emb[rows, , drop = FALSE]))
    }
    anvl::nv_array(out, dtype = "f32", device = device)
}

#' Stable Diffusion XL UNet forward (anvl)
#'
#' Returns a closure over the static SDXL configuration;
#' \code{anvl::jit()} the closure. All array arguments are dynamic
#' inputs; weights travel as the pytree from
#' \code{\link{yq_sdxl_unet_load_weights}}.
#'
#' @param block_out_channels Integer vector. Channels per block.
#' @param layers_per_block Integer. ResNet blocks per down/up block.
#' @param transformer_layers_per_block Integer vector. Transformer depth
#'   per block.
#' @param cross_attention_dim Integer. Text-encoder context width.
#' @param attention_head_dim Integer. Per-head dimension.
#'
#' @return Function of \code{(sample, t_sin, time_ids_sin, text_embeds,
#'   context, w)}:
#'   \itemize{
#'     \item sample \code{[B, 4, H, W]} noisy latents
#'     \item t_sin \code{[B, block_out_channels[1]]} sinusoidal timestep
#'       embedding (see \code{\link{yq_sdxl_time_embed}})
#'     \item time_ids_sin \code{[B, 6 * addition_time_embed_dim]}
#'       sinusoidal time-ids (see \code{\link{yq_sdxl_time_ids_embed}})
#'     \item text_embeds \code{[B, 1280]} pooled text embeds
#'     \item context \code{[B, seq, cross_attention_dim]} text embeddings
#'     \item w weights pytree (\code{\link{yq_sdxl_unet_load_weights}})
#'   }
#'   returning the predicted noise \code{[B, 4, H, W]}.
#'
#' @export
yq_sdxl_unet <- function(block_out_channels = c(320L, 640L, 1280L),
                         layers_per_block = 2L,
                         transformer_layers_per_block = c(0L, 2L, 10L),
                         cross_attention_dim = 2048L,
                         attention_head_dim = 64L) {
    cfg <- .yq_sdxl_config(block_out_channels, layers_per_block,
                           transformer_layers_per_block, attention_head_dim)
    head_dim <- cfg$head_dim

    function(sample, t_sin, time_ids_sin, text_embeds, context, w) {
        # Time-embedding MLP: linear_1 -> SiLU -> linear_2.
        temb <- yunque::yq_linear(t_sin, w$time_1_w, w$time_1_b)
        temb <- yunque::yq_silu(temb)
        temb <- yunque::yq_linear(temb, w$time_2_w, w$time_2_b)

        # Added-conditioning MLP over concat(pooled text embeds, time-ids
        # sinusoid), added to the timestep embedding.
        add_in <- anvl::nv_concatenate(text_embeds, time_ids_sin,
                                       dimension = 2L)
        add_emb <- yunque::yq_linear(add_in, w$add_1_w, w$add_1_b)
        add_emb <- yunque::yq_silu(add_emb)
        add_emb <- yunque::yq_linear(add_emb, w$add_2_w, w$add_2_b)

        emb <- temb + add_emb

        sample <- yq_conv3x3(sample, w$conv_in_w, w$conv_in_b)
        skips <- list(sample)

        for (bi in seq_along(cfg$down)) {
            bc <- cfg$down[[bi]]
            bw <- w$down[[bi]]
            for (ri in seq_len(bc$nres)) {
                sample <- .yq_sdxl_resnet(sample, emb, bw$resnets[[ri]])
                if (bc$has_attn) {
                    sample <- .yq_sdxl_spatial_transformer(
                        sample, context, bw$attentions[[ri]], bc$n_heads,
                        head_dim)
                }
                skips[[length(skips) + 1L]] <- sample
            }
            if (bc$has_down) {
                sample <- .yq_sdxl_downsample(sample, bw$downsample)
                skips[[length(skips) + 1L]] <- sample
            }
        }

        sample <- .yq_sdxl_resnet(sample, emb, w$mid$resnet1)
        sample <- .yq_sdxl_spatial_transformer(sample, context, w$mid$attn,
                                               cfg$mid_heads, head_dim)
        sample <- .yq_sdxl_resnet(sample, emb, w$mid$resnet2)

        for (bi in seq_along(cfg$up)) {
            bc <- cfg$up[[bi]]
            bw <- w$up[[bi]]
            for (ri in seq_len(bc$nres)) {
                res <- skips[[length(skips)]]
                skips[[length(skips)]] <- NULL
                sample <- anvl::nv_concatenate(sample, res, dimension = 2L)
                sample <- .yq_sdxl_resnet(sample, emb, bw$resnets[[ri]])
                if (bc$has_attn) {
                    sample <- .yq_sdxl_spatial_transformer(
                        sample, context, bw$attentions[[ri]], bc$n_heads,
                        head_dim)
                }
            }
            if (bc$has_up) {
                sample <- .yq_sdxl_upsample(sample, bw$upsample)
            }
        }

        sample <- yunque::yq_group_norm(sample, w$norm_out_w, w$norm_out_b,
                                        32L, .YQ_SDXL_EPS)
        yq_conv3x3(yunque::yq_silu(sample), w$conv_out_w, w$conv_out_b)
    }
}

#' Load SDXL UNet weights into an anvl pytree
#'
#' Reads a diffusers \code{UNet2DConditionModel} SDXL state_dict (F16/F32
#' upcast to f32) and mirrors its key tree as a nested named list for
#' \code{\link{yq_sdxl_unet}}. Conv weights load raw \code{[out, in, kH,
#' kW]} (what \code{nv_conv2d} wants); linear / attention weights
#' transpose to \code{[in, out]} for \code{\link[yunque]{yq_linear}}.
#' With \code{strict = TRUE} every checkpoint key must be consumed
#' exactly once (and no key requested that the file lacks), so a wrong
#' architecture or a missed weight fails loudly.
#'
#' @param path Path to the SDXL UNet safetensors (or the random-init
#'   state_dict fixture).
#' @param block_out_channels,layers_per_block,transformer_layers_per_block,cross_attention_dim,attention_head_dim
#'   Architecture (SDXL defaults).
#' @param device Character. Target device.
#' @param strict Logical. Enforce the exact key census.
#'
#' @return Weights pytree for \code{\link{yq_sdxl_unet}}.
#'
#' @export
yq_sdxl_unet_load_weights <- function(path,
                                      block_out_channels = c(320L, 640L, 1280L),
                                      layers_per_block = 2L,
                                      transformer_layers_per_block = c(0L, 2L, 10L),
                                      cross_attention_dim = 2048L,
                                      attention_head_dim = 64L,
                                      device = "cpu", strict = TRUE) {
    cfg <- .yq_sdxl_config(block_out_channels, layers_per_block,
                           transformer_layers_per_block, attention_head_dim)
    st <- yunque::yq_st_open(path)
    on.exit(close(st$con))
    seen <- new.env(parent = emptyenv())
    raw <- function(key) {
        assign(key, TRUE, envir = seen)
        anvl::nv_array(yunque::yq_st_read(st, key), dtype = "f32", device = device)
    }
    lin <- function(key) {
        assign(key, TRUE, envir = seen)
        anvl::nv_array(yunque::yq_st_read(st, key, transpose = TRUE),
                       dtype = "f32", device = device)
    }
    has <- function(key) !is.null(st$header[[key]])

    resnet <- function(p) {
        r <- list(
            norm1_w = raw(paste0(p, "norm1.weight")),
            norm1_b = raw(paste0(p, "norm1.bias")),
            conv1_w = raw(paste0(p, "conv1.weight")),
            conv1_b = raw(paste0(p, "conv1.bias")),
            time_emb_w = lin(paste0(p, "time_emb_proj.weight")),
            time_emb_b = raw(paste0(p, "time_emb_proj.bias")),
            norm2_w = raw(paste0(p, "norm2.weight")),
            norm2_b = raw(paste0(p, "norm2.bias")),
            conv2_w = raw(paste0(p, "conv2.weight")),
            conv2_b = raw(paste0(p, "conv2.bias"))
        )
        if (has(paste0(p, "conv_shortcut.weight"))) {
            r$shortcut_w <- raw(paste0(p, "conv_shortcut.weight"))
            r$shortcut_b <- raw(paste0(p, "conv_shortcut.bias"))
        }
        r
    }

    attn_head <- function(p) {
        list(
            to_q_w = lin(paste0(p, "to_q.weight")),
            to_k_w = lin(paste0(p, "to_k.weight")),
            to_v_w = lin(paste0(p, "to_v.weight")),
            to_out_w = lin(paste0(p, "to_out.0.weight")),
            to_out_b = raw(paste0(p, "to_out.0.bias"))
        )
    }

    tblock <- function(p) {
        list(
            norm1_w = raw(paste0(p, "norm1.weight")),
            norm1_b = raw(paste0(p, "norm1.bias")),
            attn1 = attn_head(paste0(p, "attn1.")),
            norm2_w = raw(paste0(p, "norm2.weight")),
            norm2_b = raw(paste0(p, "norm2.bias")),
            attn2 = attn_head(paste0(p, "attn2.")),
            norm3_w = raw(paste0(p, "norm3.weight")),
            norm3_b = raw(paste0(p, "norm3.bias")),
            ff = list(
                ff_in_w = lin(paste0(p, "ff.net.0.proj.weight")),
                ff_in_b = raw(paste0(p, "ff.net.0.proj.bias")),
                ff_out_w = lin(paste0(p, "ff.net.2.weight")),
                ff_out_b = raw(paste0(p, "ff.net.2.bias"))
            )
        )
    }

    spatial <- function(p, depth) {
        list(
            gn_w = raw(paste0(p, "norm.weight")),
            gn_b = raw(paste0(p, "norm.bias")),
            proj_in_w = lin(paste0(p, "proj_in.weight")),
            proj_in_b = raw(paste0(p, "proj_in.bias")),
            proj_out_w = lin(paste0(p, "proj_out.weight")),
            proj_out_b = raw(paste0(p, "proj_out.bias")),
            blocks = lapply(seq_len(depth) - 1L,
                function(i) tblock(sprintf("%stransformer_blocks.%d.", p, i)))
        )
    }

    w <- list(
        conv_in_w = raw("conv_in.weight"),
        conv_in_b = raw("conv_in.bias"),
        time_1_w = lin("time_embedding_linear_1.weight"),
        time_1_b = raw("time_embedding_linear_1.bias"),
        time_2_w = lin("time_embedding_linear_2.weight"),
        time_2_b = raw("time_embedding_linear_2.bias"),
        add_1_w = lin("add_embedding_linear_1.weight"),
        add_1_b = raw("add_embedding_linear_1.bias"),
        add_2_w = lin("add_embedding_linear_2.weight"),
        add_2_b = raw("add_embedding_linear_2.bias"),
        norm_out_w = raw("conv_norm_out.weight"),
        norm_out_b = raw("conv_norm_out.bias"),
        conv_out_w = raw("conv_out.weight"),
        conv_out_b = raw("conv_out.bias")
    )

    w$down <- lapply(seq_along(cfg$down), function(bi) {
        bc <- cfg$down[[bi]]
        p <- sprintf("down_blocks.%d.", bi - 1L)
        blk <- list(resnets = lapply(seq_len(bc$nres) - 1L,
            function(r) resnet(sprintf("%sresnets.%d.", p, r))))
        if (bc$has_attn) {
            blk$attentions <- lapply(seq_len(bc$nres) - 1L,
                function(a) spatial(sprintf("%sattentions.%d.", p, a), bc$depth))
        }
        if (bc$has_down) {
            blk$downsample <- list(
                conv_w = raw(paste0(p, "downsamplers.0.conv.weight")),
                conv_b = raw(paste0(p, "downsamplers.0.conv.bias")))
        }
        blk
    })

    w$mid <- list(
        resnet1 = resnet("mid_block.resnets.0."),
        attn = spatial("mid_block.attentions.0.", cfg$mid_depth),
        resnet2 = resnet("mid_block.resnets.1.")
    )

    w$up <- lapply(seq_along(cfg$up), function(bi) {
        bc <- cfg$up[[bi]]
        p <- sprintf("up_blocks.%d.", bi - 1L)
        blk <- list(resnets = lapply(seq_len(bc$nres) - 1L,
            function(r) resnet(sprintf("%sresnets.%d.", p, r))))
        if (bc$has_attn) {
            blk$attentions <- lapply(seq_len(bc$nres) - 1L,
                function(a) spatial(sprintf("%sattentions.%d.", p, a), bc$depth))
        }
        if (bc$has_up) {
            blk$upsample <- list(
                conv_w = raw(paste0(p, "upsamplers.0.conv.weight")),
                conv_b = raw(paste0(p, "upsamplers.0.conv.bias")))
        }
        blk
    })

    if (strict) {
        all_keys <- setdiff(names(st$header), "__metadata__")
        used <- ls(seen)
        unused <- setdiff(all_keys, used)
        extra <- setdiff(used, all_keys)
        if (length(extra)) {
            stop("SDXL UNet anvl load: ", length(extra),
                 " requested keys not in file, e.g. ",
                 paste(utils::head(extra, 5), collapse = ", "))
        }
        if (length(unused)) {
            stop("SDXL UNet anvl load: ", length(unused),
                 " checkpoint keys unused, e.g. ",
                 paste(utils::head(unused, 5), collapse = ", "))
        }
    }

    w
}
