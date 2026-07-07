#' LTX-2.3 Audio VAE
#'
#' Fresh R port of the LTX-2 audio autoencoder from the diffusers
#' reference (Apache-2.0, autoencoder_kl_ltx2_audio.py), configured per
#' the checkpoint: pixel norm, height-axis causality, base 128 channels
#' with multipliers (1, 2, 4), 8 latent channels, 64 mel bins, no
#' attention. The decoder produces mel for the vocoder; the encoder
#' turns user audio into conditioning latents (lip sync).
#'
#' @name audio_vae_ltx23
NULL

# Audio latents are 4x downsampled in time relative to mel frames
.ltx23_audio_latent_downsample_factor <- 4L

#' Causal 2D convolution for audio spectrograms
#'
#' Pads asymmetrically along the causal axis ("height" = time frames for
#' LTX audio) before an unpadded Conv2d.
#'
#' @param in_channels,out_channels Integers.
#' @param kernel_size Integer or length-2 vector.
#' @param stride Integer.
#' @param causality_axis "height", "width", "width-compatibility", or "none".
#'
#' @export
ltx23_audio_causal_conv2d <- torch::nn_module(
    "ltx23_audio_causal_conv2d",
    initialize = function(
                          in_channels,
                          out_channels,
                          kernel_size = 3L,
                          stride = 1L,
                          causality_axis = "height"
    ) {
    if (length(kernel_size) == 1L) kernel_size <- rep(kernel_size, 2L)
    pad_h <- kernel_size[1] - 1L
    pad_w <- kernel_size[2] - 1L

    self$padding <- switch(causality_axis,
                           none = c(pad_w %/% 2L, pad_w - pad_w %/% 2L, pad_h %/% 2L,
                                    pad_h - pad_h %/% 2L),
                           width = c(pad_w, 0L, pad_h %/% 2L, pad_h - pad_h %/% 2L),
                           `width-compatibility` = c(pad_w, 0L, pad_h %/% 2L, pad_h - pad_h %/% 2L),
                           height = c(pad_w %/% 2L, pad_w - pad_w %/% 2L, pad_h, 0L),
                           stop("Invalid causality_axis: ", causality_axis)
    )
    self$conv <- torch::nn_conv2d(
                                  in_channels, out_channels, kernel_size,
                                  stride = stride, padding = 0L
    )
},
    forward = function(x) {
    self$conv(torch::nnf_pad(x, self$padding))
}
)

#' LTX audio ResNet block
#'
#' PixelNorm -> SiLU -> causal conv, twice, with a 1x1 causal conv
#' shortcut (\code{nin_shortcut}) on channel change.
#'
#' @param in_channels,out_channels Integers.
#' @param causality_axis Character.
#'
#' @export
ltx23_audio_resnet_block <- torch::nn_module(
    "ltx23_audio_resnet_block",
    initialize = function(
                          in_channels,
                          out_channels = NULL,
                          causality_axis = "height"
    ) {
    out_channels <- out_channels %||% in_channels
    self$changes_channels <- in_channels != out_channels

    self$norm1 <- ltx23_per_channel_rms_norm(eps = 1e-6)
    self$conv1 <- ltx23_audio_causal_conv2d(in_channels, out_channels,
        kernel_size = 3L, causality_axis = causality_axis)
    self$norm2 <- ltx23_per_channel_rms_norm(eps = 1e-6)
    self$conv2 <- ltx23_audio_causal_conv2d(
        out_channels, out_channels, kernel_size = 3L, causality_axis = causality_axis
    )
    if (self$changes_channels) {
        self$nin_shortcut <- ltx23_audio_causal_conv2d(
            in_channels, out_channels, kernel_size = 1L, causality_axis = causality_axis
        )
    }
},
    forward = function(x) {
    h <- self$norm1(x)
    h <- torch::nnf_silu(h)
    h <- self$conv1(h)
    h <- self$norm2(h)
    h <- torch::nnf_silu(h)
    h <- self$conv2(h)
    if (self$changes_channels) {
        x <- self$nin_shortcut(x)
    }
    x + h
}
)

#' LTX audio upsampler
#'
#' Nearest 2x interpolation, causal conv, then a crop of the first
#' element along the causal axis.
#'
#' @param in_channels Integer.
#' @param causality_axis Character.
#'
#' @export
ltx23_audio_upsample <- torch::nn_module(
    "ltx23_audio_upsample",
    initialize = function(in_channels, causality_axis = "height") {
    self$causality_axis <- causality_axis
    self$conv <- ltx23_audio_causal_conv2d(in_channels, in_channels,
        kernel_size = 3L, causality_axis = causality_axis)
},
    forward = function(x) {
    x <- torch::nnf_interpolate(x, scale_factor = 2, mode = "nearest")
    x <- self$conv(x)
    if (self$causality_axis == "height") {
        # Drop the first (causally padded) frame; Python [:, :, 1:, :]
        x <- x$narrow(3L, 2L, x$shape[3] - 1L)
    } else if (self$causality_axis == "width") {
        x <- x$narrow(4L, 2L, x$shape[4] - 1L)
    }
    x
}
)

#' LTX audio downsampler
#'
#' Causal zero-pad followed by a plain stride-2 conv (reference
#' LTX2AudioDownsample; note the conv is unwrapped, so its checkpoint
#' key is \code{downsample.conv.*}).
#'
#' @param in_channels Integer.
#' @param causality_axis Character.
#'
#' @export
ltx23_audio_downsample <- torch::nn_module(
    "ltx23_audio_downsample",
    initialize = function(in_channels, causality_axis = "height") {
    # Padding order: (left, right, top, bottom)
    self$padding <- switch(causality_axis, none = c(0L, 1L, 0L, 1L),
                           width = c(2L, 0L, 0L, 1L),
                           height = c(0L, 1L, 2L, 0L),
                           `width-compatibility` = c(1L, 0L, 0L, 1L),
                           stop("Invalid causality_axis: ", causality_axis))
    self$conv <- torch::nn_conv2d(in_channels, in_channels, kernel_size = 3L,
                                  stride = 2L, padding = 0L)
},
    forward = function(x) {
    self$conv(torch::nnf_pad(x, self$padding))
}
)

#' LTX-2.3 audio VAE encoder
#'
#' Mel spectrogram [B, 2, T, 64] -> latent distribution moments
#' [B, 2 * latent_channels, ceil(T/4), 16]. Structure mirrors the
#' decoder: causal convs, parameterless pixel norms, ResNet stages with
#' stride-2 downsampling between levels (reference LTX2AudioEncoder).
#'
#' @param base_channels,num_res_blocks,latent_channels,ch_mult,causality_axis
#'   See \code{\link{ltx23_audio_decoder}}.
#' @param in_channels Integer. Mel channels (2 = stereo).
#'
#' @export
ltx23_audio_encoder <- torch::nn_module(
                                        "ltx23_audio_encoder",
                                        initialize = function(
        base_channels = 128L,
        in_channels = 2L,
        num_res_blocks = 2L,
        latent_channels = 8L,
        ch_mult = c(1L, 2L, 4L),
        causality_axis = "height"
    ) {
    num_levels <- length(ch_mult)
    self$conv_in <- ltx23_audio_causal_conv2d(in_channels, base_channels,
        kernel_size = 3L, causality_axis = causality_axis)

    block_in <- base_channels
    stages <- list()
    for (level in seq_len(num_levels)) {
        block_out <- base_channels * ch_mult[level]
        blocks <- list()
        for (j in seq_len(num_res_blocks)) {
            blocks[[j]] <- ltx23_audio_resnet_block(
                if (j == 1L) block_in else block_out, block_out,
                causality_axis = causality_axis
            )
        }
        block_in <- block_out
        stage <- torch::nn_module(
                                  "ltx23_audio_down_stage",
                                  initialize = function(blocks, downsample) {
            self$block <- torch::nn_module_list(blocks)
            if (!is.null(downsample)) {
                self$downsample <- downsample
            }
        },
                                  forward = function(x) {
            for (i in seq_along(self$block)) {
                x <- self$block[[i]](x)
            }
            if (!is.null(self$downsample)) {
                x <- self$downsample(x)
            }
            x
        }
        )
        stages[[level]] <- stage(
                                 blocks,
            if (level != num_levels) {
                ltx23_audio_downsample(block_in, causality_axis = causality_axis)
            } else {
                NULL
            }
        )
    }
    self$down <- torch::nn_module_list(stages)

    self$mid <- torch::nn_module(
                                 "ltx23_audio_mid",
                                 initialize = function(channels, causality_axis) {
        self$block_1 <- ltx23_audio_resnet_block(channels,
            causality_axis = causality_axis)
        self$block_2 <- ltx23_audio_resnet_block(channels,
            causality_axis = causality_axis)
    },
                                 forward = function(x) {
        self$block_2(self$block_1(x))
    }
    )(block_in, causality_axis)

    self$norm_out <- ltx23_per_channel_rms_norm(eps = 1e-6)
    self$conv_out <- ltx23_audio_causal_conv2d(
        block_in, 2L * latent_channels, kernel_size = 3L,
        causality_axis = causality_axis
    )
    self$latent_channels <- as.integer(latent_channels)
},
                                        forward = function(x) {
    h <- self$conv_in(x)
    for (level in seq_along(self$down)) {
        h <- self$down[[level]](h)
    }
    h <- self$mid(h)
    h <- self$norm_out(h)
    h <- torch::nnf_silu(h)
    self$conv_out(h)
}
)

#' LTX-2.3 audio VAE decoder
#'
#' Latents [B, 8, L, 16] -> mel spectrogram [B, 2, 4L - 3, 64].
#'
#' @param base_channels Integer.
#' @param output_channels Integer. Audio channels (2 = stereo).
#' @param num_res_blocks Integer. Per-level ResNet count (a stage runs
#'   \code{num_res_blocks + 1} blocks).
#' @param latent_channels Integer.
#' @param ch_mult Integer vector. Channel multipliers per level.
#' @param causality_axis Character.
#' @param mel_bins Integer. Output mel bins (crop/pad target).
#'
#' @export
ltx23_audio_decoder <- torch::nn_module(
                                        "ltx23_audio_decoder",
                                        initialize = function(
        base_channels = 128L,
        output_channels = 2L,
        num_res_blocks = 2L,
        latent_channels = 8L,
        ch_mult = c(1L, 2L, 4L),
        causality_axis = "height",
        mel_bins = 64L
    ) {
    self$num_resolutions <- length(ch_mult)
    self$num_res_blocks <- num_res_blocks
    self$out_ch <- as.integer(output_channels)
    self$mel_bins <- as.integer(mel_bins)
    self$causal <- causality_axis != "none"

    base_block_channels <- base_channels * ch_mult[length(ch_mult)]

    self$conv_in <- ltx23_audio_causal_conv2d(latent_channels,
        base_block_channels, kernel_size = 3L,
        causality_axis = causality_axis)

    mid_container <- torch::nn_module(
                                      "ltx23_audio_mid",
                                      initialize = function(channels, causality_axis) {
        self$block_1 <- ltx23_audio_resnet_block(channels, channels,
            causality_axis = causality_axis)
        self$block_2 <- ltx23_audio_resnet_block(channels, channels,
            causality_axis = causality_axis)
    },
                                      forward = function(x) {
        self$block_2(self$block_1(x))
    }
    )
    self$mid <- mid_container(base_block_channels, causality_axis)

    # Stages are stored at their level index; built from the top level
    # down, tracking the running channel count like the reference
    stages <- vector("list", self$num_resolutions)
    block_in <- base_block_channels
    for (level in rev(seq_len(self$num_resolutions))) {
        block_out <- base_channels * ch_mult[level]
        stage_blocks <- list()
        for (j in seq_len(num_res_blocks + 1L)) {
            stage_blocks[[j]] <- ltx23_audio_resnet_block(
                block_in, block_out, causality_axis = causality_axis
            )
            block_in <- block_out
        }
        stage <- torch::nn_module(
                                  "ltx23_audio_up_stage",
                                  initialize = function(blocks, upsample) {
            self$block <- torch::nn_module_list(blocks)
            if (!is.null(upsample)) self$upsample <- upsample
        },
                                  forward = function(x) {
            for (i in seq_along(self$block)) {
                x <- self$block[[i]](x)
            }
            if (!is.null(self$upsample)) x <- self$upsample(x)
            x
        }
        )
        upsample <- if (level != 1L) {
            ltx23_audio_upsample(block_in, causality_axis = causality_axis)
        } else {
            NULL
        }
        stages[[level]] <- stage(stage_blocks, upsample)
    }
    self$up <- torch::nn_module_list(stages)

    self$norm_out <- ltx23_per_channel_rms_norm(eps = 1e-6)
    self$conv_out <- ltx23_audio_causal_conv2d(
        block_in, output_channels, kernel_size = 3L, causality_axis = causality_axis
    )
},
                                        forward = function(sample) {
    frames <- sample$shape[3]
    target_frames <- frames * .ltx23_audio_latent_downsample_factor
    if (self$causal) {
        target_frames <- max(target_frames - (.ltx23_audio_latent_downsample_factor - 1L), 1L)
    }

    h <- self$conv_in(sample)
    h <- self$mid(h)
    for (level in rev(seq_len(self$num_resolutions))) {
        h <- self$up[[level]](h)
    }

    h <- self$norm_out(h)
    h <- torch::nnf_silu(h)
    h <- self$conv_out(h)

    # Crop/pad to the target (channels, time, mel bins)
    cur_time <- h$shape[3]
    cur_freq <- h$shape[4]
    h <- h$narrow(2L, 1L, self$out_ch)
    h <- h$narrow(3L, 1L, min(cur_time, target_frames))
    h <- h$narrow(4L, 1L, min(cur_freq, self$mel_bins))

    time_pad <- target_frames - h$shape[3]
    freq_pad <- self$mel_bins - h$shape[4]
    if (time_pad > 0L || freq_pad > 0L) {
        h <- torch::nnf_pad(h, c(0L, max(freq_pad, 0L), 0L, max(time_pad, 0L)))
    }
    h
}
)

#' LTX-2.3 audio VAE
#'
#' Encoder + decoder plus the per-channel latent statistics loaded from
#' the checkpoint. Encoding is used for audio-conditioned generation
#' (lip sync); decoding for generated audio.
#'
#' @param base_channels,output_channels,num_res_blocks,latent_channels,ch_mult,causality_axis,mel_bins
#'   See \code{\link{ltx23_audio_decoder}}.
#' @param in_channels Integer. Mel input channels (2 = stereo).
#'
#' @export
ltx23_audio_vae <- torch::nn_module(
                                    "ltx23_audio_vae",
                                    initialize = function(
        base_channels = 128L,
        output_channels = 2L,
        num_res_blocks = 2L,
        latent_channels = 8L,
        ch_mult = c(1L, 2L, 4L),
        causality_axis = "height",
        mel_bins = 64L,
        in_channels = 2L
    ) {
    self$encoder <- ltx23_audio_encoder(base_channels = base_channels,
                                        in_channels = in_channels,
                                        num_res_blocks = num_res_blocks,
                                        latent_channels = latent_channels,
                                        ch_mult = ch_mult,
                                        causality_axis = causality_axis)
    self$decoder <- ltx23_audio_decoder(base_channels = base_channels,
                                        output_channels = output_channels,
                                        num_res_blocks = num_res_blocks,
                                        latent_channels = latent_channels,
                                        ch_mult = ch_mult,
                                        causality_axis = causality_axis,
                                        mel_bins = mel_bins)
    # Statistics are stored at base_channels size in the checkpoint;
    # only the first latent_channels entries are meaningful
    self$latents_mean <- torch::nn_buffer(torch::torch_zeros(base_channels))
    self$latents_std <- torch::nn_buffer(torch::torch_ones(base_channels))
    self$latent_channels <- as.integer(latent_channels)
},
                                    encode = function(mel) {
    moments <- self$encoder(mel)
    n <- self$latent_channels
    list(
         mean = moments$narrow(2L, 1L, n),
         logvar = moments$narrow(2L, n + 1L, n)
    )
},
                                    decode = function(z) {
    .ltx23_traced_call(self$decoder, z)
},
                                    forward = function(z) {
    self$decode(z)
}
)

#' Map an official audio VAE checkpoint key to the R module name
#'
#' @param key Character. Checkpoint key.
#'
#' @return Character.
#'
#' @export
ltx23_map_audio_vae_key <- function(key) {
    key <- sub("^audio_vae\\.", "", key)
    key <- sub("^per_channel_statistics\\.mean-of-means$", "latents_mean", key)
    key <- sub("^per_channel_statistics\\.std-of-means$", "latents_std", key)
    key
}
