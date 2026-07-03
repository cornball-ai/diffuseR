#' LTX-2.3 Video VAE Building Blocks
#'
#' Fresh R port of the LTX-2 causal video autoencoder blocks from the
#' diffusers reference (Apache-2.0,
#' src/diffusers/models/autoencoders/autoencoder_kl_ltx2.py). Training
#' and unused inference branches (noise injection, timestep conditioning,
#' plain-conv downsampling) are intentionally not ported; the 2.3
#' checkpoints carry no such weights.
#'
#' @name vae_ltx23_modules
NULL

#' Per-channel RMS normalization
#'
#' Normalizes by the root-mean-square across the channel dimension
#' (dim 2 of [B, C, F, H, W]); no learned parameters.
#'
#' @param eps Numeric. Stability epsilon.
#'
#' @export
ltx23_per_channel_rms_norm <- torch::nn_module(
  "ltx23_per_channel_rms_norm",
  initialize = function(eps = 1e-8) {
    self$eps <- eps
  },
  forward = function(x) {
    mean_sq <- torch::torch_mean(x^2, dim = 2L, keepdim = TRUE)
    x / torch::torch_sqrt(mean_sq + self$eps)
  }
)

#' Causal 3D convolution
#'
#' Spatial padding is handled by the convolution; temporal padding
#' replicates the first frame (causal) or both edge frames (non-causal),
#' chosen at call time.
#'
#' @param in_channels,out_channels Integers.
#' @param kernel_size Integer or length-3 vector (t, h, w).
#' @param stride Integer or length-3 vector.
#' @param spatial_padding_mode Character. Conv padding mode.
#'
#' @export
ltx23_causal_conv3d <- torch::nn_module(
  "ltx23_causal_conv3d",
  initialize = function(
    in_channels,
    out_channels,
    kernel_size = 3L,
    stride = 1L,
    spatial_padding_mode = "zeros"
  ) {
    if (length(kernel_size) == 1L) kernel_size <- rep(kernel_size, 3L)
    if (length(stride) == 1L) stride <- rep(stride, 3L)
    self$kernel_size <- as.integer(kernel_size)

    height_pad <- kernel_size[2] %/% 2L
    width_pad <- kernel_size[3] %/% 2L

    self$conv <- torch::nn_conv3d(
      in_channels, out_channels, self$kernel_size,
      stride = as.integer(stride),
      padding = c(0L, height_pad, width_pad),
      padding_mode = spatial_padding_mode
    )
  },
  forward = function(hidden_states, causal = TRUE) {
    time_k <- self$kernel_size[1]
    if (time_k > 1L) {
      if (causal) {
        pad_left <- hidden_states$narrow(3L, 1L, 1L)$`repeat`(c(1L, 1L, time_k - 1L, 1L, 1L))
        hidden_states <- torch::torch_cat(list(pad_left, hidden_states), dim = 3L)
      } else {
        half <- (time_k - 1L) %/% 2L
        pad_left <- hidden_states$narrow(3L, 1L, 1L)$`repeat`(c(1L, 1L, half, 1L, 1L))
        pad_right <- hidden_states$narrow(
          3L, hidden_states$shape[3], 1L
        )$`repeat`(c(1L, 1L, half, 1L, 1L))
        hidden_states <- torch::torch_cat(list(pad_left, hidden_states, pad_right), dim = 3L)
      }
    }
    self$conv(hidden_states)
  }
)

#' LTX 3D ResNet block
#'
#' PerChannelRMSNorm -> SiLU -> causal conv, twice, with a LayerNorm +
#' 1x1 Conv3d shortcut when the channel count changes.
#'
#' @param in_channels,out_channels Integers.
#' @param eps Numeric. Shortcut LayerNorm epsilon.
#' @param spatial_padding_mode Character.
#'
#' @export
ltx23_video_resnet_block3d <- torch::nn_module(
  "ltx23_video_resnet_block3d",
  initialize = function(
    in_channels,
    out_channels = NULL,
    eps = 1e-6,
    spatial_padding_mode = "zeros"
  ) {
    out_channels <- out_channels %||% in_channels

    self$norm1 <- ltx23_per_channel_rms_norm()
    self$conv1 <- ltx23_causal_conv3d(
      in_channels, out_channels, kernel_size = 3L,
      spatial_padding_mode = spatial_padding_mode
    )
    self$norm2 <- ltx23_per_channel_rms_norm()
    self$conv2 <- ltx23_causal_conv3d(
      out_channels, out_channels, kernel_size = 3L,
      spatial_padding_mode = spatial_padding_mode
    )

    if (in_channels != out_channels) {
      self$norm3 <- torch::nn_layer_norm(in_channels, eps = eps,
        elementwise_affine = TRUE)
      # A plain (non-causal) 1x1 Conv3d, per the reference
      self$conv_shortcut <- torch::nn_conv3d(in_channels, out_channels,
        kernel_size = 1L, stride = 1L)
    }
  },
  forward = function(inputs, causal = TRUE) {
    hidden_states <- self$norm1(inputs)
    hidden_states <- torch::nnf_silu(hidden_states)
    hidden_states <- self$conv1(hidden_states, causal = causal)

    hidden_states <- self$norm2(hidden_states)
    hidden_states <- torch::nnf_silu(hidden_states)
    hidden_states <- self$conv2(hidden_states, causal = causal)

    if (!is.null(self$norm3)) {
      # LayerNorm over channels: move C last, norm, move back
      inputs <- self$norm3(inputs$permute(c(1L, 3L, 4L, 5L, 2L)))$permute(c(1L, 5L, 2L, 3L, 4L))
    }
    if (!is.null(self$conv_shortcut)) {
      inputs <- self$conv_shortcut(inputs)
    }
    hidden_states + inputs
  }
)

#' Pixel-unshuffle 3D downsampler
#'
#' Conv followed by space/time-to-channel rearrangement, plus a grouped
#' channel-mean residual of the same rearrangement.
#'
#' @param in_channels,out_channels Integers.
#' @param stride Length-3 integer vector (t, h, w).
#' @param spatial_padding_mode Character.
#'
#' @export
ltx23_video_downsampler3d <- torch::nn_module(
  "ltx23_video_downsampler3d",
  initialize = function(
    in_channels,
    out_channels,
    stride = c(1L, 1L, 1L),
    spatial_padding_mode = "zeros"
  ) {
    if (length(stride) == 1L) stride <- rep(stride, 3L)
    self$stride <- as.integer(stride)
    self$group_size <- (in_channels * prod(stride)) %/% out_channels

    conv_out <- out_channels %/% prod(stride)
    self$conv <- ltx23_causal_conv3d(
      in_channels, conv_out, kernel_size = 3L, stride = 1L,
      spatial_padding_mode = spatial_padding_mode
    )
  },
  forward = function(hidden_states, causal = TRUE) {
    s <- self$stride
    if (s[1] > 1L) {
      pad <- hidden_states$narrow(3L, 1L, 1L)$`repeat`(c(1L, 1L, s[1] - 1L, 1L, 1L))
      hidden_states <- torch::torch_cat(list(pad, hidden_states), dim = 3L)
    }

    # Space/time-to-channel rearrangement shared by both paths
    to_channels <- function(x) {
      x <- x$unflatten(5L, c(-1L, s[3]))$unflatten(4L, c(-1L, s[2]))$unflatten(3L, c(-1L, s[1]))
      # [B, C, F', st, H', sh, W', sw] -> [B, C, st, sh, sw, F', H', W']
      x$permute(c(1L, 2L, 4L, 6L, 8L, 3L, 5L, 7L))$flatten(start_dim = 2L, end_dim = 5L)
    }

    residual <- to_channels(hidden_states)
    residual <- residual$unflatten(2L, c(-1L, self$group_size))$mean(dim = 3L)

    hidden_states <- self$conv(hidden_states, causal = causal)
    hidden_states <- to_channels(hidden_states)
    hidden_states + residual
  }
)

#' Pixel-shuffle 3D upsampler
#'
#' Conv followed by channel-to-space/time rearrangement, with an optional
#' channel-repeat residual and an upscale factor that divides the conv
#' output channels.
#'
#' @param in_channels Integer.
#' @param stride Length-3 integer vector (t, h, w).
#' @param residual Logical. Add the rearranged input as a residual.
#' @param upscale_factor Integer.
#' @param spatial_padding_mode Character.
#'
#' @export
ltx23_video_upsampler3d <- torch::nn_module(
  "ltx23_video_upsampler3d",
  initialize = function(
    in_channels,
    stride = c(1L, 1L, 1L),
    residual = FALSE,
    upscale_factor = 1L,
    spatial_padding_mode = "zeros"
  ) {
    if (length(stride) == 1L) stride <- rep(stride, 3L)
    self$stride <- as.integer(stride)
    self$residual <- residual
    self$upscale_factor <- as.integer(upscale_factor)

    out_channels <- (in_channels * prod(stride)) %/% upscale_factor
    self$conv <- ltx23_causal_conv3d(
      in_channels, out_channels, kernel_size = 3L, stride = 1L,
      spatial_padding_mode = spatial_padding_mode
    )
  },
  forward = function(hidden_states, causal = TRUE) {
    s <- self$stride
    dims <- hidden_states$shape
    num_frames <- dims[3]; height <- dims[4]; width <- dims[5]

    # Channel-to-space/time rearrangement shared by both paths
    to_space <- function(x) {
      x <- x$reshape(c(dims[1], -1L, s[1], s[2], s[3], num_frames, height, width))
      # [B, C', st, sh, sw, F, H, W] -> [B, C', F, st, H, sh, W, sw]
      x <- x$permute(c(1L, 2L, 6L, 3L, 7L, 4L, 8L, 5L))
      x <- x$flatten(start_dim = 7L, end_dim = 8L)$
        flatten(start_dim = 5L, end_dim = 6L)$
        flatten(start_dim = 3L, end_dim = 4L)
      # Drop the causally duplicated leading frames
      if (s[1] > 1L) x <- x$narrow(3L, s[1], x$shape[3] - s[1] + 1L) else x
    }

    residual <- NULL
    if (self$residual) {
      residual <- to_space(hidden_states)
      repeats <- prod(s) %/% self$upscale_factor
      residual <- residual$`repeat`(c(1L, repeats, 1L, 1L, 1L))
    }

    hidden_states <- self$conv(hidden_states, causal = causal)
    hidden_states <- to_space(hidden_states)

    if (!is.null(residual)) hidden_states <- hidden_states + residual
    hidden_states
  }
)

#' LTX video down block
#'
#' ResNet stack (at the input channel count) followed by a
#' pixel-unshuffle downsampler that also changes the channel count.
#'
#' @param in_channels,out_channels Integers.
#' @param num_layers Integer. ResNet count.
#' @param resnet_eps Numeric.
#' @param spatio_temporal_scale Logical. Whether to downsample at all.
#' @param downsample_type "spatial", "temporal", or "spatiotemporal".
#' @param spatial_padding_mode Character.
#'
#' @export
ltx23_video_down_block3d <- torch::nn_module(
  "ltx23_video_down_block3d",
  initialize = function(
    in_channels,
    out_channels = NULL,
    num_layers = 1L,
    resnet_eps = 1e-6,
    spatio_temporal_scale = TRUE,
    downsample_type = "spatiotemporal",
    spatial_padding_mode = "zeros"
  ) {
    out_channels <- out_channels %||% in_channels

    self$resnets <- torch::nn_module_list(lapply(seq_len(num_layers), function(i) {
      ltx23_video_resnet_block3d(
        in_channels, in_channels, eps = resnet_eps,
        spatial_padding_mode = spatial_padding_mode
      )
    }))

    if (spatio_temporal_scale) {
      stride <- switch(downsample_type,
        spatial = c(1L, 2L, 2L),
        temporal = c(2L, 1L, 1L),
        spatiotemporal = c(2L, 2L, 2L),
        stop("Unknown downsample_type: ", downsample_type)
      )
      self$downsamplers <- torch::nn_module_list(list(
        ltx23_video_downsampler3d(
          in_channels, out_channels, stride = stride,
          spatial_padding_mode = spatial_padding_mode
        )
      ))
    }
  },
  forward = function(hidden_states, causal = TRUE) {
    for (i in seq_along(self$resnets)) {
      hidden_states <- self$resnets[[i]](hidden_states, causal = causal)
    }
    if (!is.null(self$downsamplers)) {
      for (i in seq_along(self$downsamplers)) {
        hidden_states <- self$downsamplers[[i]](hidden_states, causal = causal)
      }
    }
    hidden_states
  }
)

#' LTX video mid block
#'
#' A plain ResNet stack at a fixed channel count.
#'
#' @param in_channels Integer.
#' @param num_layers Integer.
#' @param resnet_eps Numeric.
#' @param spatial_padding_mode Character.
#'
#' @export
ltx23_video_mid_block3d <- torch::nn_module(
  "ltx23_video_mid_block3d",
  initialize = function(
    in_channels,
    num_layers = 1L,
    resnet_eps = 1e-6,
    spatial_padding_mode = "zeros"
  ) {
    self$resnets <- torch::nn_module_list(lapply(seq_len(num_layers), function(i) {
      ltx23_video_resnet_block3d(
        in_channels, in_channels, eps = resnet_eps,
        spatial_padding_mode = spatial_padding_mode
      )
    }))
  },
  forward = function(hidden_states, causal = TRUE) {
    for (i in seq_along(self$resnets)) {
      hidden_states <- self$resnets[[i]](hidden_states, causal = causal)
    }
    hidden_states
  }
)

#' LTX video up block
#'
#' Optional channel-changing conv-in ResNet, pixel-shuffle upsampler,
#' then a ResNet stack at the output channel count.
#'
#' @param in_channels,out_channels Integers.
#' @param num_layers Integer.
#' @param resnet_eps Numeric.
#' @param spatio_temporal_scale Logical.
#' @param upsample_type "spatial", "temporal", or "spatiotemporal".
#' @param upsample_residual Logical.
#' @param upscale_factor Integer.
#' @param spatial_padding_mode Character.
#'
#' @export
ltx23_video_up_block3d <- torch::nn_module(
  "ltx23_video_up_block3d",
  initialize = function(
    in_channels,
    out_channels = NULL,
    num_layers = 1L,
    resnet_eps = 1e-6,
    spatio_temporal_scale = TRUE,
    upsample_type = "spatiotemporal",
    upsample_residual = FALSE,
    upscale_factor = 1L,
    spatial_padding_mode = "zeros"
  ) {
    out_channels <- out_channels %||% in_channels

    if (in_channels != out_channels) {
      self$conv_in <- ltx23_video_resnet_block3d(
        in_channels, out_channels, eps = resnet_eps,
        spatial_padding_mode = spatial_padding_mode
      )
    }

    if (spatio_temporal_scale) {
      stride <- switch(upsample_type,
        spatial = c(1L, 2L, 2L),
        temporal = c(2L, 1L, 1L),
        spatiotemporal = c(2L, 2L, 2L),
        stop("Unknown upsample_type: ", upsample_type)
      )
      self$upsamplers <- torch::nn_module_list(list(
        ltx23_video_upsampler3d(
          in_channels = out_channels * upscale_factor,
          stride = stride,
          residual = upsample_residual,
          upscale_factor = upscale_factor,
          spatial_padding_mode = spatial_padding_mode
        )
      ))
    }

    self$resnets <- torch::nn_module_list(lapply(seq_len(num_layers), function(i) {
      ltx23_video_resnet_block3d(
        out_channels, out_channels, eps = resnet_eps,
        spatial_padding_mode = spatial_padding_mode
      )
    }))
  },
  forward = function(hidden_states, causal = TRUE) {
    if (!is.null(self$conv_in)) {
      hidden_states <- self$conv_in(hidden_states, causal = causal)
    }
    if (!is.null(self$upsamplers)) {
      for (i in seq_along(self$upsamplers)) {
        hidden_states <- self$upsamplers[[i]](hidden_states, causal = causal)
      }
    }
    for (i in seq_along(self$resnets)) {
      hidden_states <- self$resnets[[i]](hidden_states, causal = causal)
    }
    hidden_states
  }
)
