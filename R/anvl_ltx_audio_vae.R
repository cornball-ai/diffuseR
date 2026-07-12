#' LTX-2.3 audio VAE decoder (anvl port of ltx23_audio_decoder)
#'
#' anvl re-implementation of \code{diffuseR::ltx23_audio_decoder}: the
#' causal-conv audio decoder that turns packed audio latents into a mel
#' spectrogram for the vocoder (the vocoder mel->waveform stage is a
#' separate component, not ported here). Structure: \code{conv_in}; a mid
#' block of two ResNet blocks; up stages (one per channel multiplier)
#' each a stack of channel-changing ResNet blocks plus an optional
#' nearest-2x upsampler; a per-channel RMS-norm -> SiLU -> \code{conv_out}
#' head; then a crop/pad to the target (channels, time, mel bins).
#'
#' The audio VAE treats the mel spectrogram as a 2D \code{[B, C, T, M]}
#' tensor (T = time frames on the "height" axis, M = mel bins on the
#' "width" axis). It differs from the 3D video VAE
#' (\code{R/anvl_ltx_vae.R}) in three ways that matter for the port:
#' \itemize{
#'   \item \strong{Causal Conv2d with ZERO padding}, not edge replication.
#'     The time axis is padded on the leading (past) side by \code{k_t-1}
#'     zeros; the mel axis gets symmetric zero padding. Implemented as an
#'     explicit \code{\link[anvl]{nv_pad}} (mirroring the reference
#'     \code{nnf_pad}) followed by an unpadded \code{\link[anvl]{nv_conv2d}}.
#'   \item \strong{Nearest-2x + causal-conv upsampling}, not pixel shuffle.
#'     \code{\link[yunque]{yq_upsample_nearest2d}} doubles both axes, a
#'     causal conv follows, then the first (causally duplicated) time
#'     frame is dropped.
#'   \item \strong{Channel-changing ResNet blocks} with a 1x1 causal-conv
#'     \code{nin_shortcut} on the residual. Unlike the 2.3 video decoder
#'     (every ResNet in==out), each audio up stage's first block changes
#'     channels, so the shortcut branch is required.
#' }
#'
#' Per-channel RMS norm (\code{ltx23_per_channel_rms_norm}, eps 1e-6) is
#' parameterless, so the entire state_dict is Conv2d \code{weight}/
#' \code{bias} pairs.
#'
#' @name anvl_ltx_audio_vae
NULL

# ltx23 audio VAE per-channel RMS norm epsilon (the audio blocks pass
# eps = 1e-6 explicitly; the shared .yq_ltx_rms_norm default is 1e-8).
.YQ_LTX_AUDIO_RMS_EPS <- 1e-6

# Audio latents are downsampled 4x in time relative to mel frames.
.YQ_LTX_AUDIO_TIME_DOWNSAMPLE <- 4L

#' Causal 2D convolution for audio spectrograms (anvl)
#'
#' Zero-pads asymmetrically before an unpadded stride-1 Conv2d, matching
#' \code{diffuseR::ltx23_audio_causal_conv2d}. For \code{causality_axis =
#' "height"} the time axis (dim 3) is padded by \code{k_h-1} on the
#' leading side only; the mel axis (dim 4) gets symmetric padding.
#'
#' @param x AnvlArray \code{[B, C_in, T, M]}.
#' @param weight AnvlArray \code{[C_out, C_in, k_h, k_w]} (raw torch layout).
#' @param bias AnvlArray \code{[C_out]}.
#' @param causality_axis Character. "height" (default), "width", or "none".
#'
#' @return AnvlArray \code{[B, C_out, T, M]}.
#'
#' @export
yq_ltx_audio_causal_conv2d <- function(x, weight, bias,
                                       causality_axis = "height") {
    ks <- anvl::shape(weight)
    pad_h <- ks[3L] - 1L
    pad_w <- ks[4L] - 1L
    # torch nnf_pad order is (w_left, w_right, h_top, h_bottom).
    pad <- switch(causality_axis,
                  none = c(pad_w %/% 2L, pad_w - pad_w %/% 2L, pad_h %/% 2L,
                           pad_h - pad_h %/% 2L),
                  width = c(pad_w, 0L, pad_h %/% 2L, pad_h - pad_h %/% 2L),
                  `width-compatibility` = c(pad_w, 0L, pad_h %/% 2L,
            pad_h - pad_h %/% 2L),
                  height = c(pad_w %/% 2L, pad_w - pad_w %/% 2L, pad_h, 0L),
                  stop("Invalid causality_axis: ", causality_axis))
    if (pad_h > 0L || pad_w > 0L) {
        zero <- anvl::nv_scalar(0, "f32")
        # NCHW dims (B, C, H=T, W=M): low/high per dim.
        x <- anvl::nv_pad(x, zero,
                          edge_padding_low = c(0L, 0L, pad[3L], pad[1L]),
                          edge_padding_high = c(0L, 0L, pad[4L], pad[2L]))
    }
    y <- anvl::nv_conv2d(x, weight, stride = 1L, padding = 0L)
    s <- anvl::shape(y)
    y + anvl::nv_broadcast_to(anvl::nv_reshape(bias, c(1L, s[2L], 1L, 1L)), s)
}

# ResNet block: (RMS-norm -> SiLU -> causal conv) x2 + residual, with a
# 1x1 causal-conv shortcut on channel change.
.yq_ltx_audio_resnet <- function(x, w, causality_axis) {
    h <- .yq_ltx_rms_norm(x, .YQ_LTX_AUDIO_RMS_EPS)
    h <- yunque::yq_silu(h)
    h <- yq_ltx_audio_causal_conv2d(h, w$conv1_w, w$conv1_b, causality_axis)
    h <- .yq_ltx_rms_norm(h, .YQ_LTX_AUDIO_RMS_EPS)
    h <- yunque::yq_silu(h)
    h <- yq_ltx_audio_causal_conv2d(h, w$conv2_w, w$conv2_b, causality_axis)
    if (!is.null(w$shortcut_w)) {
        # kernel 1 -> no padding, plain 1x1 conv.
        x <- yq_ltx_audio_causal_conv2d(x, w$shortcut_w, w$shortcut_b,
                                        causality_axis)
    }
    h + x
}

# Upsampler: nearest-2x on both axes, causal conv, then drop the leading
# (causally duplicated) time frame.
.yq_ltx_audio_upsample <- function(x, u, causality_axis) {
    x <- yunque::yq_upsample_nearest2d(x)
    x <- yq_ltx_audio_causal_conv2d(x, u$w, u$b, causality_axis)
    s <- anvl::shape(x)
    if (causality_axis == "height") {
        x <- anvl::nv_static_slice(x, c(1L, 1L, 2L, 1L),
                                   c(s[1L], s[2L], s[3L], s[4L]),
                                   c(1L, 1L, 1L, 1L))
    } else if (causality_axis == "width") {
        x <- anvl::nv_static_slice(x, c(1L, 1L, 1L, 2L),
                                   c(s[1L], s[2L], s[3L], s[4L]),
                                   c(1L, 1L, 1L, 1L))
    }
    x
}

#' LTX-2.3 audio VAE decoder forward (anvl)
#'
#' \code{anvl::jit()} the wrapper \code{function(z) yq_ltx_audio_vae_decode(z, w)}.
#' \code{z} must be the unpacked decoder input \code{[B, C_lat, T, M]}
#' (see \code{\link{yq_ltx_audio_vae_prepare}}). All crop/pad geometry is
#' static (derived from \code{z}'s trace-time shape).
#'
#' @param z AnvlArray \code{[B, C_lat, T, M]} unpacked audio latents.
#' @param w Weights pytree from \code{\link{yq_ltx_audio_vae_load_weights}}.
#'
#' @return AnvlArray \code{[B, out_ch, 4T-3, mel_bins]} mel spectrogram
#'   (\code{4T-3} frames in causal mode).
#'
#' @export
yq_ltx_audio_vae_decode <- function(z, w) {
    ca <- w$causality_axis
    frames <- anvl::shape(z)[3L]
    target_frames <- frames * .YQ_LTX_AUDIO_TIME_DOWNSAMPLE
    if (w$causal) {
        target_frames <- max(
                             target_frames - (.YQ_LTX_AUDIO_TIME_DOWNSAMPLE - 1L), 1L)
    }

    x <- yq_ltx_audio_causal_conv2d(z, w$conv_in$w, w$conv_in$b, ca)
    x <- .yq_ltx_audio_resnet(x, w$mid$block1, ca)
    x <- .yq_ltx_audio_resnet(x, w$mid$block2, ca)
    for (blk in w$up_blocks) {
        for (r in blk$resnets) {
            x <- .yq_ltx_audio_resnet(x, r, ca)
        }
        if (!is.null(blk$upsample)) {
            x <- .yq_ltx_audio_upsample(x, blk$upsample, ca)
        }
    }
    x <- .yq_ltx_rms_norm(x, .YQ_LTX_AUDIO_RMS_EPS)
    x <- yunque::yq_silu(x)
    x <- yq_ltx_audio_causal_conv2d(x, w$conv_out$w, w$conv_out$b, ca)

    # Crop/pad to (out_ch, target_frames, mel_bins).
    s <- anvl::shape(x)
    time_keep <- min(s[3L], target_frames)
    freq_keep <- min(s[4L], w$mel_bins)
    x <- anvl::nv_static_slice(x, c(1L, 1L, 1L, 1L),
                               c(s[1L], w$out_ch, time_keep, freq_keep),
                               c(1L, 1L, 1L, 1L))
    time_pad <- target_frames - time_keep
    freq_pad <- w$mel_bins - freq_keep
    if (time_pad > 0L || freq_pad > 0L) {
        zero <- anvl::nv_scalar(0, "f32")
        x <- anvl::nv_pad(x, zero,
                          edge_padding_low = c(0L, 0L, 0L, 0L),
                          edge_padding_high = c(0L, 0L, time_pad, freq_pad))
    }
    x
}

#' Prepare packed audio latents for the VAE decoder (de-normalize + unpack)
#'
#' Mirrors the pipeline's pre-decode step (\code{.ltx23_denormalize_audio}
#' then \code{ltx23_unpack_audio_latents}): applies the per-channel affine
#' \code{z * std + mean} on the packed \code{[B, T, C*M]} representation
#' (stats broadcast over the trailing feature dim), then reshapes to the
#' decoder's \code{[B, C, T, M]} input. The decoder's own \code{decode()}
#' applies no further normalization.
#'
#' @param z_packed AnvlArray \code{[B, T, C*M]} packed audio latents.
#' @param latents_mean,latents_std Numeric vectors \code{[C*M]}.
#' @param num_mel_bins Integer. Latent mel bins \code{M} (\code{C*M} splits
#'   into \code{(C, M)} row-major).
#'
#' @return AnvlArray \code{[B, C, T, M]}, de-normalized and unpacked.
#'
#' @export
yq_ltx_audio_vae_prepare <- function(z_packed, latents_mean, latents_std,
                                     num_mel_bins) {
    dev <- anvl::device(z_packed)
    s <- anvl::shape(z_packed)
    b <- s[1L]
    t <- s[2L]
    cm <- s[3L]
    mean_a <- anvl::nv_array(array(as.double(latents_mean), c(1L, 1L, cm)),
                             dtype = "f32", device = dev)
    std_a <- anvl::nv_array(array(as.double(latents_std), c(1L, 1L, cm)),
                            dtype = "f32", device = dev)
    z <- z_packed * anvl::nv_broadcast_to(std_a, s) +
    anvl::nv_broadcast_to(mean_a, s)
    # unpack: [B, T, C*M] -> [B, T, C, M] -> [B, C, T, M]
    n_mel <- as.integer(num_mel_bins)
    z <- anvl::nv_reshape(z, c(b, t, cm %/% n_mel, n_mel))
    anvl::nv_transpose(z, c(1L, 3L, 2L, 4L))
}

#' Load LTX-2.3 audio VAE decoder weights into an anvl pytree
#'
#' Reads a \code{diffuseR::ltx23_audio_decoder} state_dict (F16/F32 upcast
#' to f32): \code{conv_in}, the two mid ResNet blocks, the up stages (each
#' a ResNet stack whose first block carries a \code{nin_shortcut}, plus an
#' optional \code{upsample} conv), and \code{conv_out}. The up stages are
#' returned in forward-processing order (top level first, i.e.
#' \code{up.<n-1>} down to \code{up.0}). The per-channel RMS norms carry no
#' weights, so the whole census is Conv2d \code{weight}/\code{bias} pairs.
#' Conv weights load raw \code{[out, in, k_h, k_w]}. Output channels are
#' read from \code{conv_out}. With \code{strict = TRUE} every file key must
#' be consumed exactly once.
#'
#' @param path Path to the decoder state_dict \code{.safetensors} (native
#'   \code{ltx23_audio_decoder} key names, no \code{decoder.} prefix).
#' @param device Character. Target device.
#' @param strict Logical. Enforce the exact key census.
#' @param mel_bins Integer. Output mel bins (crop/pad target).
#' @param causality_axis Character. Padding convention ("height" default).
#'
#' @return Weights pytree for \code{\link{yq_ltx_audio_vae_decode}}.
#'
#' @export
yq_ltx_audio_vae_load_weights <- function(path, device = "cpu",
    strict = TRUE, mel_bins = 64L,
    causality_axis = "height") {
    st <- yunque::yq_st_open(path)
    on.exit(close(st$con))
    seen <- new.env(parent = emptyenv())
    raw <- function(key) {
        assign(key, TRUE, envir = seen)
        anvl::nv_array(yunque::yq_st_read(st, key), dtype = "f32",
                       device = device)
    }
    has <- function(key) !is.null(st$header[[key]])

    conv <- function(p) {
        list(w = raw(paste0(p, ".conv.weight")),
             b = raw(paste0(p, ".conv.bias")))
    }
    resnet <- function(p) {
        r <- list(
                  conv1_w = raw(paste0(p, "conv1.conv.weight")),
                  conv1_b = raw(paste0(p, "conv1.conv.bias")),
                  conv2_w = raw(paste0(p, "conv2.conv.weight")),
                  conv2_b = raw(paste0(p, "conv2.conv.bias"))
        )
        if (has(paste0(p, "nin_shortcut.conv.weight"))) {
            r$shortcut_w <- raw(paste0(p, "nin_shortcut.conv.weight"))
            r$shortcut_b <- raw(paste0(p, "nin_shortcut.conv.bias"))
        }
        r
    }
    count_blocks <- function(prefix) {
        n <- 0L
        while (has(sprintf("%sblock.%d.conv1.conv.weight", prefix, n))) {
            n <- n + 1L
        }
        n
    }

    n_up <- 0L
    while (has(sprintf("up.%d.block.0.conv1.conv.weight", n_up))) {
        n_up <- n_up + 1L
    }

    # Forward order runs the top level first: up.(n_up-1) .. up.0.
    up_blocks <- lapply(rev(seq_len(n_up) - 1L), function(i) {
        bp <- sprintf("up.%d.", i)
        nb <- count_blocks(bp)
        blk <- list(resnets = lapply(seq_len(nb) - 1L,
                                     function(j) resnet(sprintf("%sblock.%d.", bp, j))))
        if (has(paste0(bp, "upsample.conv.conv.weight"))) {
            blk$upsample <- conv(paste0(bp, "upsample.conv"))
        }
        blk
    })

    w <- list(
              conv_in = conv("conv_in"),
              conv_out = conv("conv_out"),
              mid = list(block1 = resnet("mid.block_1."),
                         block2 = resnet("mid.block_2.")),
              up_blocks = up_blocks,
              causality_axis = causality_axis,
              causal = causality_axis != "none",
              mel_bins = as.integer(mel_bins)
    )
    w$out_ch <- anvl::shape(w$conv_out$w)[1L]

    if (strict) {
        all_keys <- setdiff(names(st$header), "__metadata__")
        used <- ls(seen)
        unused <- setdiff(all_keys, used)
        extra <- setdiff(used, all_keys)
        if (length(extra)) {
            stop("LTX audio VAE anvl load: ", length(extra),
                 " requested keys not in file, e.g. ",
                 paste(utils::head(extra, 5), collapse = ", "))
        }
        if (length(unused)) {
            stop("LTX audio VAE anvl load: ", length(unused),
                 " keys unused, e.g. ",
                 paste(utils::head(unused, 5), collapse = ", "))
        }
    }

    w
}
