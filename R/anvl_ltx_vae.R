#' LTX-2.3 3D video VAE decoder (anvl port of ltx23_video_decoder3d)
#'
#' anvl re-implementation of \code{diffuseR::ltx23_video_decoder3d}: the
#' causal-conv video decoder (\code{conv_in}; a mid block of ResNet
#' blocks; four up blocks each an optional channel-changing ResNet, a
#' pixel-shuffle upsampler, and a ResNet stack; a per-channel RMS-norm
#' -> SiLU -> \code{conv_out} head; channel-to-space un-patchification).
#' Decodes \code{[B, C_lat, F, H, W]} latents to
#' \code{[B, 3, 8F-7, 32H, 32W]} pixels in [-1, 1].
#'
#' Structural differences from the 2D AutoencoderKL decoders (SD/FLUX/
#' Z-Image, \code{R/anvl_vae.R}):
#' \itemize{
#'   \item \strong{Causal Conv3d} not Conv2d. The time axis is padded by
#'     \emph{edge replication} (repeat the first/last frame), not zeros:
#'     causal mode replicates the first frame \code{k_t-1} times on the
#'     left; non-causal mode (the decoder default) replicates the first
#'     frame \code{(k_t-1)/2} times on the left and the last frame the
#'     same on the right. The spatial axes get symmetric zero padding.
#'     Built with \code{\link[anvl]{prim_convolution}} on the NCDHW
#'     dimension numbers plus a padding matrix whose temporal row is
#'     \code{c(0, 0)} (already replicated) and spatial rows are
#'     \code{c(k//2, k//2)}.
#'   \item \strong{Per-channel RMS norm} not GroupNorm. Normalizes each
#'     spatiotemporal position by the root-mean-square across the channel
#'     dim, no learned parameters (\code{ltx23_per_channel_rms_norm}).
#'   \item \strong{Pixel-shuffle up/downsampling} not nearest-neighbour.
#'     The upsampler convs, then rearranges channels into space/time
#'     (\code{to_space}), dropping the causally duplicated leading frames
#'     when it upsamples time.
#'   \item \strong{Per-channel latent de-normalization} (latents_mean/std),
#'     applied host-side by \code{\link{yq_ltx_vae_prepare}}.
#' }
#'
#' The 2.3 decoder has no ResNet shortcuts (every ResNet is in==out) and
#' no upsample residuals, so those branches are intentionally not ported.
#'
#' @name anvl_ltx_vae
NULL

# ltx23_per_channel_rms_norm default eps.
.YQ_LTX_RMS_EPS <- 1e-8

# NCDHW convolution dimension numbers for prim_convolution: batch=1,
# feature=2, spatial=3:5 on both the [B,C,F,H,W] input and the
# [out,in,kT,kH,kW] kernel (anvl 1-based dims).
.yq_ltx_conv_dn <- function() {
    list(input_batch_dimension = 1L, input_feature_dimension = 2L,
         input_spatial_dimensions = c(3L, 4L, 5L),
         kernel_output_feature_dimension = 1L,
         kernel_input_feature_dimension = 2L,
         kernel_spatial_dimensions = c(3L, 4L, 5L),
         output_batch_dimension = 1L, output_feature_dimension = 2L,
         output_spatial_dimensions = c(3L, 4L, 5L))
}

# Slice a single frame `idx` (1-based) along the temporal dim 3, keeping
# [B, C, 1, H, W].
.yq_ltx_frame <- function(x, idx) {
    s <- anvl::shape(x)
    anvl::nv_static_slice(x, c(1L, 1L, as.integer(idx), 1L, 1L),
                          c(s[1L], s[2L], as.integer(idx), s[4L], s[5L]),
                          c(1L, 1L, 1L, 1L, 1L))
}

#' Causal 3D convolution (anvl)
#'
#' Temporal edge-replication padding then a stride-1 Conv3d with
#' symmetric spatial zero padding, via \code{\link[anvl]{prim_convolution}}.
#' Matches \code{diffuseR::ltx23_causal_conv3d}'s forward for both the
#' causal (left-only) and non-causal (symmetric) temporal padding modes.
#'
#' @param x AnvlArray \code{[B, C_in, F, H, W]}.
#' @param weight AnvlArray \code{[C_out, C_in, k_t, k_h, k_w]} (raw torch
#'   layout).
#' @param bias AnvlArray \code{[C_out]}.
#' @param causal Logical. Left-only temporal padding when \code{TRUE};
#'   symmetric when \code{FALSE} (the decoder default).
#' @param kernel_t Integer. Temporal kernel size (3 in LTX; padding is
#'   skipped when 1).
#'
#' @return AnvlArray \code{[B, C_out, F, H, W]}.
#'
#' @export
yq_ltx_causal_conv3d <- function(x, weight, bias, causal = FALSE,
                                 kernel_t = 3L) {
    kernel_t <- as.integer(kernel_t)
    if (kernel_t > 1L) {
        s <- anvl::shape(x)
        first <- .yq_ltx_frame(x, 1L)
        if (causal) {
            reps <- rep(list(first), kernel_t - 1L)
            x <- do.call(anvl::nv_concatenate,
                         c(reps, list(x), list(dimension = 3L)))
        } else {
            half <- (kernel_t - 1L) %/% 2L
            last <- .yq_ltx_frame(x, s[3L])
            x <- do.call(anvl::nv_concatenate, c(
                    rep(list(first), half), list(x), rep(list(last), half),
                    list(dimension = 3L)
                ))
        }
    }
    # Kernel spatial padding = (k-1)/2 symmetric; temporal already handled.
    ks <- anvl::shape(weight)
    hp <- ks[4L] %/% 2L
    wp <- ks[5L] %/% 2L
    y <- anvl::prim_convolution(
                                x, weight,
                                dimension_numbers = .yq_ltx_conv_dn(),
                                window_strides = c(1L, 1L, 1L),
                                padding = rbind(c(0L, 0L), c(hp, hp), c(wp, wp)),
                                lhs_dilation = c(1L, 1L, 1L), rhs_dilation = c(1L, 1L, 1L),
                                feature_group_count = 1L, batch_group_count = 1L
    )
    sy <- anvl::shape(y)
    y + anvl::nv_broadcast_to(anvl::nv_reshape(bias, c(1L, sy[2L], 1L, 1L, 1L)), sy)
}

# Plain 1x1x1 Conv3d (no padding), for ResNet shortcuts. Kept for
# generality; the 2.3 decoder never triggers it.
.yq_ltx_conv1x1 <- function(x, weight, bias) {
    y <- anvl::prim_convolution(
                                x, weight,
                                dimension_numbers = .yq_ltx_conv_dn(),
                                window_strides = c(1L, 1L, 1L),
                                padding = rbind(c(0L, 0L), c(0L, 0L), c(0L, 0L)),
                                lhs_dilation = c(1L, 1L, 1L), rhs_dilation = c(1L, 1L, 1L),
                                feature_group_count = 1L, batch_group_count = 1L
    )
    sy <- anvl::shape(y)
    y + anvl::nv_broadcast_to(anvl::nv_reshape(bias, c(1L, sy[2L], 1L, 1L, 1L)),
                              sy)
}

# Per-channel RMS norm over the channel dim (dim 2), no learned params.
.yq_ltx_rms_norm <- function(x, eps = .YQ_LTX_RMS_EPS) {
    ms <- anvl::nv_mean(x * x, dims = 2L, drop = FALSE)
    denom <- anvl::nv_sqrt(ms + eps)
    x / anvl::nv_broadcast_to(denom, anvl::shape(x))
}

# ResNet block: (RMS-norm -> SiLU -> causal conv) x2 + residual. The 2.3
# decoder has no channel-changing ResNet, so the LayerNorm+1x1 shortcut
# is only wired defensively.
.yq_ltx_resnet <- function(x, w, causal) {
    h <- .yq_ltx_rms_norm(x)
    h <- yunque::yq_silu(h)
    h <- yq_ltx_causal_conv3d(h, w$conv1_w, w$conv1_b, causal)
    h <- .yq_ltx_rms_norm(h)
    h <- yunque::yq_silu(h)
    h <- yq_ltx_causal_conv3d(h, w$conv2_w, w$conv2_b, causal)
    if (!is.null(w$shortcut_w)) {
        x <- .yq_ltx_conv1x1(x, w$shortcut_w, w$shortcut_b)
    }
    h + x
}

# Pixel-shuffle upsampler: causal conv then channel-to-space/time
# rearrangement (dropping the causally duplicated leading frames when the
# temporal stride > 1). The 2.3 decoder sets residual = FALSE throughout.
.yq_ltx_upsampler <- function(x, w, causal) {
    s <- anvl::shape(x)
    b <- s[1L]; nf <- s[3L]; ht <- s[4L]; wd <- s[5L]
    st <- w$stride[1L]; sh <- w$stride[2L]; sw <- w$stride[3L]

    h <- yq_ltx_causal_conv3d(x, w$conv_w, w$conv_b, causal)
    co <- anvl::shape(h)[2L]
    cp <- co %/% (st * sh * sw)
    # [B, co, F, H, W] -> [B, C', st, sh, sw, F, H, W]
    h <- anvl::nv_reshape(h, c(b, cp, st, sh, sw, nf, ht, wd))
    # -> [B, C', F, st, H, sh, W, sw]
    h <- anvl::nv_transpose(h, c(1L, 2L, 6L, 3L, 7L, 4L, 8L, 5L))
    # -> [B, C', F*st, H*sh, W*sw]
    h <- anvl::nv_reshape(h, c(b, cp, nf * st, ht * sh, wd * sw))
    if (st > 1L) {
        sf <- anvl::shape(h)
        h <- anvl::nv_static_slice(h, c(1L, 1L, st, 1L, 1L),
                                   c(sf[1L], sf[2L], sf[3L], sf[4L], sf[5L]),
                                   c(1L, 1L, 1L, 1L, 1L))
    }
    h
}

# Up block: optional channel-changing ResNet, optional upsampler, ResNet
# stack.
.yq_ltx_up_block <- function(x, blk, causal) {
    if (!is.null(blk$conv_in)) {
        x <- .yq_ltx_resnet(x, blk$conv_in, causal)
    }
    if (!is.null(blk$upsampler)) {
        x <- .yq_ltx_upsampler(x, blk$upsampler, causal)
    }
    for (r in blk$resnets) {
        x <- .yq_ltx_resnet(x, r, causal)
    }
    x
}

# Channel-to-space un-patchification (LTX ordering), matching the decoder
# forward tail: [B, C*pt*p*p, F, H, W] -> [B, C, F*pt, H*p, W*p].
.yq_ltx_unpatchify <- function(x, p_t, p) {
    s <- anvl::shape(x)
    b <- s[1L]; nf <- s[3L]; ht <- s[4L]; wd <- s[5L]
    cout <- s[2L] %/% (p_t * p * p)
    # [B, C, pt, ph, pw, F, H, W]
    x <- anvl::nv_reshape(x, c(b, cout, p_t, p, p, nf, ht, wd))
    # -> [B, C, F, pt, H, pw, W, ph]
    x <- anvl::nv_transpose(x, c(1L, 2L, 6L, 3L, 7L, 5L, 8L, 4L))
    # -> [B, C, F*pt, H*pw, W*ph]
    anvl::nv_reshape(x, c(b, cout, nf * p_t, ht * p, wd * p))
}

#' LTX-2.3 VAE decoder forward (anvl)
#'
#' \code{anvl::jit()} the wrapper \code{function(z) yq_ltx_vae_decode(z, w)}.
#' \code{z} must already be latent de-normalized (see
#' \code{\link{yq_ltx_vae_prepare}}). \code{causal} is a static (trace-time)
#' branch selector; the LTX decoder runs non-causal (\code{FALSE}).
#'
#' @param z AnvlArray \code{[B, C_lat, F, H, W]} de-normalized latents.
#' @param w Weights pytree from \code{\link{yq_ltx_vae_load_weights}}.
#' @param causal Logical. Temporal padding mode (decoder default FALSE).
#'
#' @return AnvlArray \code{[B, 3, 8F-7, 32H, 32W]} pixels in [-1, 1].
#'
#' @export
yq_ltx_vae_decode <- function(z, w, causal = FALSE) {
    x <- yq_ltx_causal_conv3d(z, w$conv_in_w, w$conv_in_b, causal)
    for (r in w$mid) {
        x <- .yq_ltx_resnet(x, r, causal)
    }
    for (blk in w$up_blocks) {
        x <- .yq_ltx_up_block(x, blk, causal)
    }
    x <- .yq_ltx_rms_norm(x)
    x <- yunque::yq_silu(x)
    x <- yq_ltx_causal_conv3d(x, w$conv_out_w, w$conv_out_b, causal)
    .yq_ltx_unpatchify(x, w$patch_size_t, w$patch_size)
}

#' Prepare LTX latents for the VAE decoder (per-channel de-normalization)
#'
#' Applies the inverse of the VAE's per-channel latent normalization,
#' \code{z * std + mean} broadcast over \code{[B, C, F, H, W]} (mirrors
#' \code{diffuseR::ltx23_denormalize_latents}). The stats are supplied as
#' host-side R vectors; the affine runs device-side so \code{z} stays on
#' the device. The checkpoint's \code{scaling_factor} is 1.0, so this
#' per-channel affine is the whole de-normalization.
#'
#' @param z AnvlArray \code{[B, C, F, H, W]} normalized latents.
#' @param latents_mean,latents_std Numeric vectors \code{[C]}.
#'
#' @return AnvlArray \code{[B, C, F, H, W]}, de-normalized.
#'
#' @export
yq_ltx_vae_prepare <- function(z, latents_mean, latents_std) {
    dev <- anvl::device(z)
    n <- length(latents_mean)
    mean_a <- anvl::nv_array(array(as.double(latents_mean),
                                   c(1L, n, 1L, 1L, 1L)),
                             dtype = "f32", device = dev)
    std_a <- anvl::nv_array(array(as.double(latents_std), c(1L, n, 1L, 1L, 1L)),
                            dtype = "f32", device = dev)
    s <- anvl::shape(z)
    z * anvl::nv_broadcast_to(std_a, s) + anvl::nv_broadcast_to(mean_a, s)
}

#' Load LTX-2.3 VAE decoder weights into an anvl pytree
#'
#' Reads a \code{diffuseR::ltx23_video_decoder3d} state_dict (F16/F32
#' upcast to f32): \code{conv_in}, the mid ResNet stack, the four up
#' blocks (each an optional channel-changing ResNet \code{conv_in}, an
#' optional \code{upsamplers.0} pixel-shuffle conv, and a ResNet stack),
#' and \code{conv_out}. The per-channel RMS norms carry no weights, so the
#' entire census is Conv3d \code{weight}/\code{bias} pairs. Conv weights
#' load raw \code{[out, in, k_t, k_h, k_w]} (what
#' \code{\link[anvl]{prim_convolution}} expects). Resnet counts and up
#' block count are probed by key existence; the fixed LTX-2.3 up-block
#' pixel-shuffle geometry (strides \code{(2,2,2)}, \code{(2,2,2)},
#' \code{(2,1,1)}, \code{(1,2,2)}; upscale factors \code{2, 1, 2, 2}) is
#' hard-coded. With \code{strict = TRUE} every file key must be consumed
#' exactly once.
#'
#' @param path Path to the decoder state_dict \code{.safetensors} (native
#'   \code{ltx23_video_decoder3d} key names, no \code{decoder.} prefix).
#' @param device Character. Target device.
#' @param strict Logical. Enforce the exact key census.
#' @param patch_size,patch_size_t Integers. Pixel un-patchification factors
#'   (LTX 2.3 defaults 4 and 1).
#'
#' @return Weights pytree for \code{\link{yq_ltx_vae_decode}}.
#'
#' @export
yq_ltx_vae_load_weights <- function(path, device = "cpu", strict = TRUE,
                                    patch_size = 4L, patch_size_t = 1L) {
    st <- yunque::yq_st_open(path)
    on.exit(close(st$con))
    seen <- new.env(parent = emptyenv())
    raw <- function(key) {
        assign(key, TRUE, envir = seen)
        anvl::nv_array(yunque::yq_st_read(st, key), dtype = "f32",
                       device = device)
    }
    has <- function(key) !is.null(st$header[[key]])

    resnet <- function(p) {
        r <- list(
                  conv1_w = raw(paste0(p, "conv1.conv.weight")),
                  conv1_b = raw(paste0(p, "conv1.conv.bias")),
                  conv2_w = raw(paste0(p, "conv2.conv.weight")),
                  conv2_b = raw(paste0(p, "conv2.conv.bias"))
        )
        if (has(paste0(p, "conv_shortcut.weight"))) {
            r$shortcut_w <- raw(paste0(p, "conv_shortcut.weight"))
            r$shortcut_b <- raw(paste0(p, "conv_shortcut.bias"))
            # norm3 (LayerNorm) also present; loaded but the port errors
            # if reached, since the 2.3 decoder never has a shortcut.
            r$norm3_w <- raw(paste0(p, "norm3.weight"))
            r$norm3_b <- raw(paste0(p, "norm3.bias"))
            stop("yq_ltx_vae_load_weights: ResNet shortcut present; the ",
                 "LTX-2.3 decoder has none and the anvl port omits that branch")
        }
        r
    }
    count_resnets <- function(prefix) {
        n <- 0L
        while (has(sprintf("%sresnets.%d.conv1.conv.weight", prefix, n))) {
            n <- n + 1L
        }
        n
    }

    w <- list(
              conv_in_w = raw("conv_in.conv.weight"),
              conv_in_b = raw("conv_in.conv.bias"),
              conv_out_w = raw("conv_out.conv.weight"),
              conv_out_b = raw("conv_out.conv.bias"),
              patch_size = as.integer(patch_size),
              patch_size_t = as.integer(patch_size_t)
    )

    nm <- count_resnets("mid_block.")
    w$mid <- lapply(seq_len(nm) - 1L,
                    function(j) resnet(sprintf("mid_block.resnets.%d.", j)))

    # Fixed LTX-2.3 decoder up-block pixel-shuffle geometry.
    up_stride <- list(c(2L, 2L, 2L), c(2L, 2L, 2L), c(2L, 1L, 1L), c(1L, 2L, 2L))
    up_upscale <- c(2L, 1L, 2L, 2L)
    n_up <- 0L
    while (has(sprintf("up_blocks.%d.upsamplers.0.conv.conv.weight", n_up)) ||
        has(sprintf("up_blocks.%d.resnets.0.conv1.conv.weight", n_up))) {
        n_up <- n_up + 1L
    }

    w$up_blocks <- lapply(seq_len(n_up) - 1L, function(i) {
        bp <- sprintf("up_blocks.%d.", i)
        blk <- list()
        if (has(paste0(bp, "conv_in.conv1.conv.weight"))) {
            blk$conv_in <- resnet(paste0(bp, "conv_in."))
        }
        if (has(paste0(bp, "upsamplers.0.conv.conv.weight"))) {
            blk$upsampler <- list(
                                  conv_w = raw(paste0(bp, "upsamplers.0.conv.conv.weight")),
                                  conv_b = raw(paste0(bp, "upsamplers.0.conv.conv.bias")),
                                  stride = up_stride[[i + 1L]],
                                  upscale = up_upscale[i + 1L]
            )
        }
        nr <- count_resnets(bp)
        blk$resnets <- lapply(seq_len(nr) - 1L,
                              function(j) resnet(sprintf("%sresnets.%d.", bp, j)))
        blk
    })

    if (strict) {
        all_keys <- setdiff(names(st$header), "__metadata__")
        used <- ls(seen)
        unused <- setdiff(all_keys, used)
        extra <- setdiff(used, all_keys)
        if (length(extra)) {
            stop("LTX VAE anvl load: ", length(extra),
                 " requested keys not in file, e.g. ",
                 paste(utils::head(extra, 5), collapse = ", "))
        }
        if (length(unused)) {
            stop("LTX VAE anvl load: ", length(unused),
                 " keys unused, e.g. ",
                 paste(utils::head(unused, 5), collapse = ", "))
        }
    }

    w
}
