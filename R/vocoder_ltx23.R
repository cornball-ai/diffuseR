#' LTX-2.3 Vocoder with Bandwidth Extension
#'
#' Fresh R port of the LTX-2 BigVGAN-style vocoder from the diffusers
#' reference (Apache-2.0, pipelines/ltx2/vocoder.py). The 2.3 vocoder
#' runs a 16 kHz stage (hidden 1536, snakebeta activations with
#' anti-aliased up/downsampling), re-analyzes its output into a causal
#' log-mel spectrogram, and feeds a bandwidth-extension vocoder whose
#' residual is added to a Hann-resampled skip path for 48 kHz output.
#' The Kaiser sinc / Hann filters and STFT bases are checkpoint buffers.
#' Runs in float32 (small model; snakebeta is precision-sensitive).
#'
#' @name vocoder_ltx23
NULL

# Kaiser window (periodic = FALSE), base-R besselI implementation used
# when building filters at init; checkpoint buffers override them.
.ltx23_kaiser_window <- function(n, beta) {
    if (n == 1L) {
        return(torch::torch_ones(1L))
    }
    k <- seq(0L, n - 1L)
    ratio <- (2 * k / (n - 1)) - 1
    vals <- besselI(beta * sqrt(pmax(1 - ratio ^ 2, 0)), 0) / besselI(beta, 0)
    torch::torch_tensor(vals, dtype = torch::torch_float32())
}

#' Kaiser sinc low-pass filter kernel
#'
#' @param cutoff Numeric. Normalized cutoff in (0, 0.5].
#' @param half_width Numeric. Transition band half width.
#' @param kernel_size Integer.
#'
#' @return Tensor [kernel_size].
#'
#' @export
ltx23_kaiser_sinc_filter1d <- function(cutoff, half_width, kernel_size) {
    delta_f <- 4 * half_width
    half_size <- kernel_size %/% 2L
    amplitude <- 2.285 * (half_size - 1) * pi * delta_f + 7.95
    beta <- if (amplitude > 50) {
        0.1102 * (amplitude - 8.7)
    } else if (amplitude >= 21) {
        0.5842 * (amplitude - 21) ^ 0.4 + 0.07886 * (amplitude - 21)
    } else {
        0
    }

    window <- .ltx23_kaiser_window(kernel_size, beta)

    even <- kernel_size %% 2L == 0L
    time <- if (even) {
        torch::torch_arange(start = -half_size, end = half_size - 1,
                            dtype = torch::torch_float32()) + 0.5
    } else {
        torch::torch_arange(start = 0, end = kernel_size - 1,
                            dtype = torch::torch_float32()) - half_size
    }

    if (cutoff == 0) {
        return(torch::torch_zeros_like(time))
    }
    time <- 2 * cutoff * time
    sinc <- torch::torch_where(
                               time == 0,
                               torch::torch_ones_like(time),
                               torch::torch_sin(pi * time) / pi / time
    )
    filter <- 2 * cutoff * window * sinc
    filter / filter$sum()
}

#' Anti-aliasing 1D downsampler (low-pass then stride)
#'
#' @param ratio Integer. Downsampling ratio.
#' @param kernel_size Integer or NULL (default 6*ratio rounded even).
#'
#' @return Module whose forward(x) returns \code{x} low-pass filtered
#'   and decimated by \code{ratio} along the time axis.
#'
#' @export
ltx23_downsample1d <- torch::nn_module(
                                       "ltx23_downsample1d",
                                       initialize = function(ratio = 2L, kernel_size = NULL) {
    self$ratio <- as.integer(ratio)
    self$kernel_size <- as.integer(kernel_size %||% (as.integer(6 * ratio %/% 2) * 2L))
    self$pad_left <- self$kernel_size %/% 2L + (self$kernel_size %% 2L) - 1L
    self$pad_right <- self$kernel_size %/% 2L

    lp <- ltx23_kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, self$kernel_size)
    self$filter <- torch::nn_buffer(lp$view(c(1L, 1L, self$kernel_size)))
},
                                       forward = function(x) {
    num_channels <- x$shape[2]
    x <- torch::nnf_pad(x, c(self$pad_left, self$pad_right), mode = "replicate")
    torch::nnf_conv1d(x, self$filter$expand(c(num_channels, -1L, -1L)),
                      stride = self$ratio, groups = num_channels)
}
)

#' Anti-aliasing 1D upsampler (transposed low-pass)
#'
#' @param ratio Integer. Upsampling ratio.
#' @param kernel_size Integer or NULL.
#' @param window_type "kaiser" (BigVGAN default) or "hann" (final resampler).
#' @param persistent Logical. Register the filter as a buffer (present in
#'   checkpoints); FALSE stores the computed filter as a plain field.
#'
#' @return Module whose forward(x) returns \code{x} interpolated up by
#'   \code{ratio} along the time axis, with the filter padding trimmed
#'   off.
#'
#' @export
ltx23_upsample1d <- torch::nn_module(
                                     "ltx23_upsample1d",
                                     initialize = function(ratio = 2L, kernel_size = NULL, window_type = "kaiser",
        persistent = TRUE) {
    self$ratio <- as.integer(ratio)

    if (window_type == "hann") {
        rolloff <- 0.99
        lowpass_filter_width <- 6L
        width <- as.integer(ceiling(lowpass_filter_width / rolloff))
        self$kernel_size <- 2L * width * self$ratio + 1L
        self$pad <- width
        self$pad_left <- 2L * width * self$ratio
        self$pad_right <- self$kernel_size - self$ratio

        time_axis <- (torch::torch_arange(start = 0,
                end = self$kernel_size - 1, dtype = torch::torch_float32()) / ratio - width) * rolloff
        time_clamped <- time_axis$clamp(-lowpass_filter_width, lowpass_filter_width)
        window <- torch::torch_cos(time_clamped * pi / lowpass_filter_width / 2) ^ 2
        # sinc(x) = sin(pi x) / (pi x), 1 at x = 0
        sinc <- torch::torch_where(
                                   time_axis == 0,
                                   torch::torch_ones_like(time_axis),
                                   torch::torch_sin(pi * time_axis) / (pi * time_axis)
        )
        filt <- (sinc * window * rolloff / ratio)$view(c(1L, 1L, -1L))
    } else {
        self$kernel_size <- as.integer(kernel_size %||% (as.integer(6 * ratio %/% 2) * 2L))
        self$pad <- self$kernel_size %/% self$ratio - 1L
        self$pad_left <- self$pad * self$ratio + (self$kernel_size - self$ratio) %/% 2L
        self$pad_right <- self$pad * self$ratio + (self$kernel_size - self$ratio + 1L) %/% 2L
        filt <- ltx23_kaiser_sinc_filter1d(
            0.5 / ratio, 0.6 / ratio, self$kernel_size
        )$view(c(1L, 1L, -1L))
    }
    if (persistent) {
        self$filter <- torch::nn_buffer(filt)
    } else {
        # Computed filter, absent from checkpoints (moved at use time)
        self$filter <- filt
    }
},
                                     forward = function(x) {
    num_channels <- x$shape[2]
    x <- torch::nnf_pad(x, c(self$pad, self$pad), mode = "replicate")
    lp <- self$filter$to(dtype = x$dtype, device = x$device)$expand(c(num_channels, -1L, -1L))
    x <- torch::nnf_conv_transpose1d(x, lp, stride = self$ratio,
                                     groups = num_channels)$mul(self$ratio)
    n <- x$shape[3]
    x$narrow(3L, self$pad_left + 1L, n - self$pad_left - self$pad_right)
}
)

#' SnakeBeta activation
#'
#' \code{x + (1 / (beta + eps)) * sin(x * alpha)^2} with per-channel
#' log-scale alpha/beta parameters.
#'
#' @param channels Integer.
#' @param eps Numeric.
#'
#' @return Module whose forward(hidden_states) returns the Snake
#'   activation \code{x + sin(alpha * x)^2 / beta}, a tensor of the same
#'   shape as the input.
#'
#' @export
ltx23_snake_beta <- torch::nn_module(
                                     "ltx23_snake_beta",
                                     initialize = function(channels, eps = 1e-9) {
    self$eps <- eps
    self$alpha <- torch::nn_parameter(torch::torch_zeros(channels))
    self$beta <- torch::nn_parameter(torch::torch_zeros(channels))
},
                                     forward = function(hidden_states) {
    shape <- rep(1L, hidden_states$ndim)
    shape[2] <- -1L
    alpha <- torch::torch_exp(self$alpha$view(shape))
    beta <- torch::torch_exp(self$beta$view(shape))
    hidden_states + torch::torch_sin(hidden_states * alpha)$pow(2) *
    beta$add(self$eps)$reciprocal()
}
)

#' Anti-aliased activation
#'
#' Upsample 2x, apply the activation, downsample 2x.
#'
#' @param channels Integer. Channels for the SnakeBeta activation.
#' @param ratio,kernel_size Integers. Resampling config.
#'
#' @return Module whose forward(x) returns the activation applied at 2x
#'   rate (upsample, activate, downsample), a tensor of the same shape
#'   as \code{x}, with the aliasing the raw activation would introduce
#'   filtered out.
#'
#' @export
ltx23_antialias_act1d <- torch::nn_module(
    "ltx23_antialias_act1d",
    initialize = function(channels, ratio = 2L, kernel_size = 12L) {
    self$upsample <- ltx23_upsample1d(ratio = ratio, kernel_size = kernel_size)
    self$act <- ltx23_snake_beta(channels)
    self$downsample <- ltx23_downsample1d(ratio = ratio,
        kernel_size = kernel_size)
},
    forward = function(x) {
    self$downsample(self$act(self$upsample(x)))
}
)

#' Vocoder ResNet block (AMP)
#'
#' Dilated conv pairs, each preceded by an anti-aliased SnakeBeta
#' activation, with residual connections.
#'
#' @param channels Integer.
#' @param kernel_size Integer.
#' @param dilations Integer vector.
#' @param antialias_ratio,antialias_kernel_size Integers.
#'
#' @return Module whose forward(x) returns \code{x} after the dilated
#'   convolution pairs have been added back as residuals, a tensor of
#'   the same shape.
#'
#' @export
ltx23_vocoder_resblock <- torch::nn_module(
    "ltx23_vocoder_resblock",
    initialize = function(
                          channels,
                          kernel_size = 3L,
                          dilations = c(1L, 3L, 5L),
                          antialias_ratio = 2L,
                          antialias_kernel_size = 12L
    ) {
    self$convs1 <- torch::nn_module_list(lapply(dilations, function(d) {
        torch::nn_conv1d(channels, channels, kernel_size, stride = 1L,
                         dilation = d, padding = d * (kernel_size - 1L) %/% 2L)
    }))
    self$acts1 <- torch::nn_module_list(lapply(dilations, function(d) {
        ltx23_antialias_act1d(channels, antialias_ratio, antialias_kernel_size)
    }))
    self$convs2 <- torch::nn_module_list(lapply(dilations, function(d) {
        torch::nn_conv1d(channels, channels, kernel_size, stride = 1L,
                         dilation = 1L, padding = (kernel_size - 1L) %/% 2L)
    }))
    self$acts2 <- torch::nn_module_list(lapply(dilations, function(d) {
        ltx23_antialias_act1d(channels, antialias_ratio, antialias_kernel_size)
    }))
},
    forward = function(x) {
    for (i in seq_along(self$convs1)) {
        xt <- self$acts1[[i]](x)
        xt <- self$convs1[[i]](xt)
        xt <- self$acts2[[i]](xt)
        xt <- self$convs2[[i]](xt)
        x <- x + xt
    }
    x
}
)

#' LTX-2.3 vocoder stage
#'
#' Mel spectrogram [B, C, T, M] -> waveform [B, out_channels, samples].
#' Channel and mel dims are flattened into conv channels; each upsample
#' stage halves the channel count and averages three parallel ResNet
#' branches.
#'
#' @param in_channels Integer. Flattened input channels (C * mel bins / 1).
#' @param hidden_channels Integer.
#' @param out_channels Integer.
#' @param upsample_kernel_sizes,upsample_factors Integer vectors.
#' @param resnet_kernel_sizes Integer vector.
#' @param resnet_dilations List of integer vectors.
#' @param antialias_ratio,antialias_kernel_size Integers.
#' @param final_bias Logical.
#'
#' @return Module whose forward(hidden_states, time_last) returns the
#'   synthesized waveform [B, 1, samples] for a mel spectrogram.
#'
#' @export
ltx23_vocoder <- torch::nn_module(
                                  "ltx23_vocoder",
                                  initialize = function(
        in_channels = 128L,
        hidden_channels = 1536L,
        out_channels = 2L,
        upsample_kernel_sizes = c(11L, 4L, 4L, 4L, 4L, 4L),
        upsample_factors = c(5L, 2L, 2L, 2L, 2L, 2L),
        resnet_kernel_sizes = c(3L, 7L, 11L),
        resnet_dilations = list(c(1L, 3L, 5L), c(1L, 3L, 5L), c(1L, 3L, 5L)),
        antialias_ratio = 2L,
        antialias_kernel_size = 12L,
        final_bias = FALSE
    ) {
    self$num_upsample_layers <- length(upsample_kernel_sizes)
    self$resnets_per_upsample <- length(resnet_kernel_sizes)

    self$conv_in <- torch::nn_conv1d(in_channels, hidden_channels,
                                     kernel_size = 7L, stride = 1L,
                                     padding = 3L)

    upsamplers <- list()
    resnets <- list()
    input_channels <- hidden_channels
    for (i in seq_along(upsample_factors)) {
        output_channels <- input_channels %/% 2L
        upsamplers[[i]] <- torch::nn_conv_transpose1d(
            input_channels, output_channels, upsample_kernel_sizes[i],
            stride = upsample_factors[i],
            padding = (upsample_kernel_sizes[i] - upsample_factors[i]) %/% 2L
        )
        for (j in seq_along(resnet_kernel_sizes)) {
            resnets[[length(resnets) + 1L]] <- ltx23_vocoder_resblock(
                channels = output_channels,
                kernel_size = resnet_kernel_sizes[j],
                dilations = resnet_dilations[[j]],
                antialias_ratio = antialias_ratio,
                antialias_kernel_size = antialias_kernel_size
            )
        }
        input_channels <- output_channels
    }
    self$upsamplers <- torch::nn_module_list(upsamplers)
    self$resnets <- torch::nn_module_list(resnets)

    self$act_out <- ltx23_antialias_act1d(output_channels, antialias_ratio,
        antialias_kernel_size)
    self$conv_out <- torch::nn_conv1d(output_channels, out_channels, 7L,
                                      stride = 1L, padding = 3L, bias = final_bias)
},
                                  forward = function(hidden_states, time_last = FALSE) {
    if (!time_last) {
        hidden_states <- hidden_states$transpose(3L, 4L)
    }
    hidden_states <- hidden_states$flatten(start_dim = 2L, end_dim = 3L)

    hidden_states <- self$conv_in(hidden_states)
    for (i in seq_len(self$num_upsample_layers)) {
        hidden_states <- self$upsamplers[[i]](hidden_states)
        start <- (i - 1L) * self$resnets_per_upsample
        branch_outputs <- lapply(seq_len(self$resnets_per_upsample), function(j) {
            self$resnets[[start + j]](hidden_states)
        })
        hidden_states <- torch::torch_mean(
            torch::torch_stack(branch_outputs, dim = 1L), dim = 1L
        )
    }
    hidden_states <- self$act_out(hidden_states)
    self$conv_out(hidden_states)
}
)

# Causal STFT with checkpoint-loaded DFT bases
ltx23_causal_stft <- torch::nn_module(
                                      "ltx23_causal_stft",
                                      initialize = function(filter_length = 512L, hop_length = 80L, window_length = 512L) {
    self$hop_length <- as.integer(hop_length)
    self$window_length <- as.integer(window_length)
    n_freqs <- filter_length %/% 2L + 1L
    self$forward_basis <- torch::nn_buffer(torch::torch_zeros(n_freqs * 2L,
            1L, filter_length))
    self$inverse_basis <- torch::nn_buffer(torch::torch_zeros(n_freqs * 2L, 1L, filter_length))
},
                                      forward = function(waveform) {
    if (waveform$ndim == 2L) waveform <- waveform$unsqueeze(2L)
    left_pad <- max(0L, self$window_length - self$hop_length)
    waveform <- torch::nnf_pad(waveform, c(left_pad, 0L))

    spec <- torch::nnf_conv1d(waveform, self$forward_basis, stride = self$hop_length)
    n_freqs <- spec$shape[2] %/% 2L
    real <- spec$narrow(2L, 1L, n_freqs)
    imag <- spec$narrow(2L, n_freqs + 1L, n_freqs)
    magnitude <- torch::torch_sqrt(real ^ 2 + imag ^ 2)
    list(magnitude = magnitude)
}
)

#' Causal log-mel spectrogram with checkpoint-loaded bases
#'
#' @param filter_length,hop_length,window_length,num_mel_channels Integers.
#'
#' @return Module whose forward(waveform) returns the log-mel
#'   spectrogram [B, n_mels, frames], clamped at 1e-5 before the log.
#'
#' @export
ltx23_mel_stft <- torch::nn_module(
                                   "ltx23_mel_stft",
                                   initialize = function(
        filter_length = 512L,
        hop_length = 80L,
        window_length = 512L,
        num_mel_channels = 64L
    ) {
    self$stft_fn <- ltx23_causal_stft(filter_length, hop_length, window_length)
    num_freqs <- filter_length %/% 2L + 1L
    self$mel_basis <- torch::nn_buffer(torch::torch_zeros(num_mel_channels,
            num_freqs))
},
                                   forward = function(waveform) {
    magnitude <- self$stft_fn(waveform)$magnitude
    mel <- torch::torch_matmul(self$mel_basis$to(dtype = magnitude$dtype), magnitude)
    torch::torch_log(torch::torch_clamp(mel, min = 1e-5))
}
)

#' LTX-2.3 vocoder with bandwidth extension
#'
#' Full mel [B, 2, T, 64] -> 48 kHz stereo waveform pipeline: 16 kHz
#' vocoder, causal mel re-analysis, BWE vocoder residual added to a
#' Hann-resampled skip path, clamped to [-1, 1].
#'
#' @param in_channels,bwe_in_channels Integers. Flattened mel input channels.
#' @param hidden_channels,bwe_hidden_channels Integers.
#' @param out_channels Integer. Audio channels.
#' @param upsample_kernel_sizes,upsample_factors,bwe_upsample_kernel_sizes,bwe_upsample_factors
#'   Integer vectors. Per-stage transposed-conv configs.
#' @param resnet_kernel_sizes,bwe_resnet_kernel_sizes Integer vectors.
#' @param resnet_dilations,bwe_resnet_dilations Lists of integer vectors.
#' @param filter_length,window_length,num_mel_channels Integers. Mel
#'   re-analysis configuration.
#' @param input_sampling_rate,output_sampling_rate Integers.
#' @param hop_length Integer. Mel analysis hop.
#'
#' @return Module whose forward(mel_spec) returns the
#'   bandwidth-extended waveform [B, 1, samples], trimmed to the sample
#'   count implied by the input frames and the rate ratio.
#'
#' @export
ltx23_vocoder_with_bwe <- torch::nn_module(
    "ltx23_vocoder_with_bwe",
    initialize = function(
                          in_channels = 128L,
                          hidden_channels = 1536L,
                          out_channels = 2L,
                          upsample_kernel_sizes = c(11L, 4L, 4L, 4L, 4L, 4L),
                          upsample_factors = c(5L, 2L, 2L, 2L, 2L, 2L),
                          resnet_kernel_sizes = c(3L, 7L, 11L),
                          resnet_dilations = NULL,
                          bwe_in_channels = 128L,
                          bwe_hidden_channels = 512L,
                          bwe_upsample_kernel_sizes = c(12L, 11L, 4L, 4L, 4L),
                          bwe_upsample_factors = c(6L, 5L, 2L, 2L, 2L),
                          bwe_resnet_kernel_sizes = c(3L, 7L, 11L),
                          bwe_resnet_dilations = NULL,
                          filter_length = 512L,
                          hop_length = 80L,
                          window_length = 512L,
                          num_mel_channels = 64L,
                          input_sampling_rate = 16000L,
                          output_sampling_rate = 48000L
    ) {
    if (is.null(resnet_dilations)) {
        resnet_dilations <- list(c(1L, 3L, 5L), c(1L, 3L, 5L), c(1L, 3L, 5L))
    }
    if (is.null(bwe_resnet_dilations)) {
        bwe_resnet_dilations <- list(c(1L, 3L, 5L), c(1L, 3L, 5L),
                                     c(1L, 3L, 5L))
    }
    self$hop_length <- as.integer(hop_length)
    self$rate_ratio <- as.integer(output_sampling_rate %/% input_sampling_rate)

    self$vocoder <- ltx23_vocoder(
                                  in_channels = in_channels,
                                  hidden_channels = hidden_channels,
                                  out_channels = out_channels,
                                  upsample_kernel_sizes = upsample_kernel_sizes,
                                  upsample_factors = upsample_factors,
                                  resnet_kernel_sizes = resnet_kernel_sizes,
                                  resnet_dilations = resnet_dilations
    )
    self$bwe_generator <- ltx23_vocoder(
                                        in_channels = bwe_in_channels,
                                        hidden_channels = bwe_hidden_channels,
                                        out_channels = out_channels,
                                        upsample_kernel_sizes = bwe_upsample_kernel_sizes,
                                        upsample_factors = bwe_upsample_factors,
                                        resnet_kernel_sizes = bwe_resnet_kernel_sizes,
                                        resnet_dilations = bwe_resnet_dilations
    )
    self$mel_stft <- ltx23_mel_stft(
                                    filter_length = filter_length,
                                    hop_length = hop_length,
                                    window_length = window_length,
                                    num_mel_channels = num_mel_channels
    )
    self$resampler <- ltx23_upsample1d(ratio = self$rate_ratio, window_type = "hann",
                                       persistent = FALSE)
},
    forward = function(mel_spec) {
    x <- self$vocoder(mel_spec)
    num_channels <- x$shape[2]
    num_samples <- x$shape[3]

    remainder <- num_samples %% self$hop_length
    if (remainder != 0L) {
        x <- torch::nnf_pad(x, c(0L, self$hop_length - remainder))
    }

    mel <- self$mel_stft(x$flatten(start_dim = 1L, end_dim = 2L))
    mel <- mel$unflatten(1L, c(-1L, num_channels))

    # [B, C, mel_bins, frames] -> [B, C, frames, mel_bins]
    residual <- self$bwe_generator(mel$transpose(3L, 4L))

    skip <- self$resampler(x)
    waveform <- torch::torch_clamp(residual + skip, -1, 1)
    output_samples <- num_samples * self$rate_ratio
    waveform$narrow(3L, 1L, min(output_samples, waveform$shape[3]))
}
)

#' Map an official vocoder checkpoint key to the R module name
#'
#' @param key Character. Checkpoint key.
#'
#' @return Character. Module parameter/buffer name.
#'
#' @export
ltx23_map_vocoder_key <- function(key) {
    key <- sub("^vocoder\\.", "", key)
    key <- gsub("resblocks", "resnets", key, fixed = TRUE)
    key <- gsub("conv_pre", "conv_in", key, fixed = TRUE)
    key <- gsub("conv_post", "conv_out", key, fixed = TRUE)
    key <- gsub("act_post", "act_out", key, fixed = TRUE)
    key <- gsub("downsample.lowpass", "downsample", key, fixed = TRUE)
    key <- gsub(".ups.", ".upsamplers.", key, fixed = TRUE)
    key <- sub("^ups\\.", "upsamplers.", key)
    key
}
