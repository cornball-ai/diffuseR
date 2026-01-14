#' LTX2 Video VAE Module Building Blocks
#'
#' Low-level nn_module components for the LTX2 3D causal VAE.
#' Used by vae_ltx2.R encoder/decoder.
#'
#' @name vae_ltx2_modules
NULL

#' Per-channel RMS normalization
#'
#' Normalizes tensor by root-mean-square along the channel dimension:
#' y = x / sqrt(mean(x^2, dim=channel_dim, keepdim=TRUE) + eps)
#'
#' @param channel_dim Integer. Dimension for RMS computation (1-indexed). Default: 2 (channels)
#' @param eps Numeric. Small constant for numerical stability. Default: 1e-8
#' @export
per_channel_rms_norm <- torch::nn_module(

"PerChannelRMSNorm",

  initialize = function(channel_dim = 2L, eps = 1e-8) {
    self$channel_dim <- channel_dim
    self$eps <- eps
  },

  forward = function(x, channel_dim = NULL) {
    if (is.null(channel_dim)) {
      channel_dim <- self$channel_dim
    }
    mean_sq <- torch::torch_mean(x^2, dim = channel_dim, keepdim = TRUE)
    rms <- torch::torch_sqrt(mean_sq + self$eps)
    x / rms
  }
)

#' LTX2 Video Causal 3D Convolution
#'
#' 3D convolution with runtime-selectable causal or non-causal padding.
#' Causal mode pads temporally by repeating first frame.
#' Non-causal mode pads temporally by repeating first and last frames.
#'
#' @param in_channels Integer. Input channels.
#' @param out_channels Integer. Output channels.
#' @param kernel_size Integer or vector of 3. Convolution kernel size.
#' @param stride Integer or vector of 3. Stride.
#' @param dilation Integer or vector of 3. Dilation.
#' @param groups Integer. Convolution groups. Default: 1
#' @param spatial_padding_mode Character. Padding mode for spatial dims. Default: "zeros"
#' @export
ltx2_video_causal_conv3d <- torch::nn_module(
  "LTX2VideoCausalConv3d",

  initialize = function(
    in_channels,
    out_channels,
    kernel_size = 3L,
    stride = 1L,
    dilation = 1L,
    groups = 1L,
    spatial_padding_mode = "zeros"
  ) {
    self$in_channels <- in_channels
    self$out_channels <- out_channels

    # Normalize kernel_size to tuple of 3
    if (length(kernel_size) == 1) {
      self$kernel_size <- rep(as.integer(kernel_size), 3)
    } else {
      self$kernel_size <- as.integer(kernel_size)
    }

    # Normalize dilation (default temporal dilation only)
    if (length(dilation) == 1) {
      dilation <- c(as.integer(dilation), 1L, 1L)
    }

    # Normalize stride
    if (length(stride) == 1) {
      stride <- rep(as.integer(stride), 3)
    }

    # Spatial padding (no temporal padding in the conv itself)
    height_pad <- self$kernel_size[2] %/% 2L
    width_pad <- self$kernel_size[3] %/% 2L
    padding <- c(0L, height_pad, width_pad)

    self$conv <- torch::nn_conv3d(
      in_channels = in_channels,
      out_channels = out_channels,
      kernel_size = self$kernel_size,
      stride = stride,
      dilation = dilation,
      groups = groups,
      padding = padding,
      padding_mode = spatial_padding_mode
    )
  },

  forward = function(hidden_states, causal = TRUE) {
    time_kernel_size <- self$kernel_size[1]

    if (causal) {
      # Causal: pad by repeating first frame on left
      pad_left <- hidden_states[, , 1:1, , ]$`repeat`(c(1L, 1L, time_kernel_size - 1L, 1L, 1L))
      hidden_states <- torch::torch_cat(list(pad_left, hidden_states), dim = 3)
    } else {
      # Non-causal: pad by repeating first/last frames symmetrically
      pad_amount <- (time_kernel_size - 1L) %/% 2L
      if (pad_amount > 0) {
        pad_left <- hidden_states[, , 1:1, , ]$`repeat`(c(1L, 1L, pad_amount, 1L, 1L))
        n_frames <- hidden_states$shape[3]
        pad_right <- hidden_states[, , n_frames:n_frames, , ]$`repeat`(c(1L, 1L, pad_amount, 1L, 1L))
        hidden_states <- torch::torch_cat(list(pad_left, hidden_states, pad_right), dim = 3)
      }
    }

    self$conv(hidden_states)
  }
)

#' LTX2 Video ResNet Block 3D
#'
#' 3D ResNet block with per-channel RMS normalization and optional
#' noise injection and timestep conditioning.
#'
#' @param in_channels Integer. Input channels.
#' @param out_channels Integer or NULL. Output channels (defaults to in_channels).
#' @param dropout Numeric. Dropout rate. Default: 0.0
#' @param eps Numeric. Epsilon for normalization. Default: 1e-6
#' @param non_linearity Character. Activation function. Default: "silu"
#' @param inject_noise Logical. Whether to inject noise. Default: FALSE
#' @param timestep_conditioning Logical. Whether to use timestep conditioning. Default: FALSE
#' @param spatial_padding_mode Character. Padding mode. Default: "zeros"
#' @export
ltx2_video_resnet_block3d <- torch::nn_module(
  "LTX2VideoResnetBlock3d",

  initialize = function(
    in_channels,
    out_channels = NULL,
    dropout = 0.0,
    eps = 1e-6,
    non_linearity = "silu",
    inject_noise = FALSE,
    timestep_conditioning = FALSE,
    spatial_padding_mode = "zeros"
  ) {
    if (is.null(out_channels)) out_channels <- in_channels

    # Activation
    self$nonlinearity <- if (non_linearity == "silu" || non_linearity == "swish") {
      torch::nn_silu()
    } else if (non_linearity == "gelu") {
      torch::nn_gelu()
    } else {
      torch::nn_relu()
    }

    self$norm1 <- per_channel_rms_norm()
    self$conv1 <- ltx2_video_causal_conv3d(
      in_channels = in_channels,
      out_channels = out_channels,
      kernel_size = 3L,
      spatial_padding_mode = spatial_padding_mode
    )

    self$norm2 <- per_channel_rms_norm()
    self$dropout <- torch::nn_dropout(p = dropout)
    self$conv2 <- ltx2_video_causal_conv3d(
      in_channels = out_channels,
      out_channels = out_channels,
      kernel_size = 3L,
      spatial_padding_mode = spatial_padding_mode
    )

    # Shortcut connection if channels differ
    self$norm3 <- NULL
    self$conv_shortcut <- NULL
    if (in_channels != out_channels) {
      self$norm3 <- torch::nn_layer_norm(in_channels, eps = eps, elementwise_affine = TRUE)
      # Regular Conv3d for shortcut (not causal)
      self$conv_shortcut <- torch::nn_conv3d(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = 1L,
        stride = 1L
      )
    }

    # Noise injection (optional)
    self$per_channel_scale1 <- NULL
    self$per_channel_scale2 <- NULL
    if (inject_noise) {
      self$per_channel_scale1 <- torch::nn_parameter(torch::torch_zeros(c(in_channels, 1, 1)))
      self$per_channel_scale2 <- torch::nn_parameter(torch::torch_zeros(c(in_channels, 1, 1)))
    }

    # Timestep conditioning (optional)
    self$scale_shift_table <- NULL
    if (timestep_conditioning) {
      init_table <- torch::torch_randn(c(4, in_channels)) / sqrt(in_channels)
      self$scale_shift_table <- torch::nn_parameter(init_table)
    }

    self$in_channels <- in_channels
    self$out_channels <- out_channels
  },

  forward = function(inputs, temb = NULL, generator = NULL, causal = TRUE) {
    hidden_states <- inputs

    hidden_states <- self$norm1(hidden_states)

    # Timestep conditioning (shift/scale)
    shift_1 <- NULL
    scale_1 <- NULL
    shift_2 <- NULL
    scale_2 <- NULL
    if (!is.null(self$scale_shift_table) && !is.null(temb)) {
      # temb shape: [B, C*4, 1, 1, 1] -> unflatten to [B, 4, C, 1, 1, 1]
      temb <- temb$unflatten(2, c(4, -1)) + self$scale_shift_table[NULL, .., NULL, NULL, NULL]
      splits <- temb$unbind(dim = 2)
      shift_1 <- splits[[1]]
      scale_1 <- splits[[2]]
      shift_2 <- splits[[3]]
      scale_2 <- splits[[4]]
      hidden_states <- hidden_states * (1 + scale_1) + shift_1
    }

    hidden_states <- self$nonlinearity(hidden_states)
    hidden_states <- self$conv1(hidden_states, causal = causal)

    # Noise injection after conv1
    if (!is.null(self$per_channel_scale1)) {
      h <- hidden_states$shape[4]
      w <- hidden_states$shape[5]
      spatial_noise <- torch::torch_randn(c(h, w), device = hidden_states$device,
                                          dtype = hidden_states$dtype)$unsqueeze(1)
      hidden_states <- hidden_states + (spatial_noise * self$per_channel_scale1)[NULL, , NULL, , ]
    }

    hidden_states <- self$norm2(hidden_states)

    # Second timestep conditioning
    if (!is.null(self$scale_shift_table) && !is.null(temb)) {
      hidden_states <- hidden_states * (1 + scale_2) + shift_2
    }

    hidden_states <- self$nonlinearity(hidden_states)
    hidden_states <- self$dropout(hidden_states)
    hidden_states <- self$conv2(hidden_states, causal = causal)

    # Noise injection after conv2
    if (!is.null(self$per_channel_scale2)) {
      h <- hidden_states$shape[4]
      w <- hidden_states$shape[5]
      spatial_noise <- torch::torch_randn(c(h, w), device = hidden_states$device,
                                          dtype = hidden_states$dtype)$unsqueeze(1)
      hidden_states <- hidden_states + (spatial_noise * self$per_channel_scale2)[NULL, , NULL, , ]
    }

    # Shortcut connection
    if (!is.null(self$norm3)) {
      # LayerNorm expects last dim to be features, so move channels to last
      inputs <- inputs$movedim(2, -1)
      inputs <- self$norm3(inputs)
      inputs <- inputs$movedim(-1, 2)
    }

    if (!is.null(self$conv_shortcut)) {
      inputs <- self$conv_shortcut(inputs)
    }

    hidden_states <- hidden_states + inputs
    hidden_states
  }
)

#' LTX Video Downsampler 3D
#'
#' Spatiotemporal downsampling with strided pixel unshuffle + convolution.
#'
#' @param in_channels Integer. Input channels.
#' @param out_channels Integer. Output channels.
#' @param stride Integer or vector of 3. Downsampling stride.
#' @param spatial_padding_mode Character. Padding mode.
#' @export
ltx_video_downsampler3d <- torch::nn_module(
  "LTXVideoDownsampler3d",

  initialize = function(
    in_channels,
    out_channels,
    stride = 1L,
    spatial_padding_mode = "zeros"
  ) {
    if (length(stride) == 1) {
      self$stride <- rep(as.integer(stride), 3)
    } else {
      self$stride <- as.integer(stride)
    }

    # Calculate group size for averaging residual
    stride_prod <- self$stride[1] * self$stride[2] * self$stride[3]
    self$group_size <- (in_channels * stride_prod) %/% out_channels

    # Output channels after pixel unshuffle
    conv_out_channels <- out_channels %/% stride_prod

    self$conv <- ltx2_video_causal_conv3d(
      in_channels = in_channels,
      out_channels = conv_out_channels,
      kernel_size = 3L,
      stride = 1L,
      spatial_padding_mode = spatial_padding_mode
    )
  },

  forward = function(hidden_states, causal = TRUE) {
    # Pad temporal dimension to handle stride
    # cat with first (stride[1]-1) frames repeated
    if (self$stride[1] > 1) {
      pad_frames <- hidden_states[, , 1:(self$stride[1] - 1), , ]
      hidden_states <- torch::torch_cat(list(pad_frames, hidden_states), dim = 3)
    }

    # Compute residual via pixel unshuffle pattern
    # unflatten spatial dims, permute, flatten to channel-like
    residual <- hidden_states$unflatten(5, c(-1, self$stride[3]))  # width
    residual <- residual$unflatten(4, c(-1, self$stride[2]))       # height
    residual <- residual$unflatten(3, c(-1, self$stride[1]))       # frames

    # Permute: [B, C, F//s, s, H//s, s, W//s, s] -> [B, C, s, s, s, F//s, H//s, W//s]
    residual <- residual$permute(c(1, 2, 4, 6, 8, 3, 5, 7))
    residual <- residual$flatten(start_dim = 2, end_dim = 5)       # [B, C*s^3, F', H', W']
    residual <- residual$unflatten(2, c(-1, self$group_size))      # [B, out_c, group_size, F', H', W']
    residual <- residual$mean(dim = 3)                             # Average over groups

    # Convolution path
    hidden_states <- self$conv(hidden_states, causal = causal)

    # Same pixel unshuffle for conv output
    hidden_states <- hidden_states$unflatten(5, c(-1, self$stride[3]))
    hidden_states <- hidden_states$unflatten(4, c(-1, self$stride[2]))
    hidden_states <- hidden_states$unflatten(3, c(-1, self$stride[1]))
    hidden_states <- hidden_states$permute(c(1, 2, 4, 6, 8, 3, 5, 7))
    hidden_states <- hidden_states$flatten(start_dim = 2, end_dim = 5)

    hidden_states <- hidden_states + residual
    hidden_states
  }
)

#' LTX Video Upsampler 3D
#'
#' Spatiotemporal upsampling with pixel shuffle + optional residual.
#'
#' @param in_channels Integer. Input channels.
#' @param stride Integer or vector of 3. Upsampling stride.
#' @param residual Logical. Whether to use residual connection.
#' @param upscale_factor Integer. Channel upscale factor.
#' @param spatial_padding_mode Character. Padding mode.
#' @export
ltx_video_upsampler3d <- torch::nn_module(
  "LTXVideoUpsampler3d",

  initialize = function(
    in_channels,
    stride = 1L,
    residual = FALSE,
    upscale_factor = 1L,
    spatial_padding_mode = "zeros"
  ) {
    if (length(stride) == 1) {
      self$stride <- rep(as.integer(stride), 3)
    } else {
      self$stride <- as.integer(stride)
    }
    self$residual <- residual
    self$upscale_factor <- upscale_factor

    stride_prod <- self$stride[1] * self$stride[2] * self$stride[3]
    out_channels <- (in_channels * stride_prod) %/% upscale_factor

    self$conv <- ltx2_video_causal_conv3d(
      in_channels = in_channels,
      out_channels = out_channels,
      kernel_size = 3L,
      stride = 1L,
      spatial_padding_mode = spatial_padding_mode
    )
  },

  forward = function(hidden_states, causal = TRUE) {
    batch_size <- hidden_states$shape[1]
    num_channels <- hidden_states$shape[2]
    num_frames <- hidden_states$shape[3]
    height <- hidden_states$shape[4]
    width <- hidden_states$shape[5]

    stride_prod <- self$stride[1] * self$stride[2] * self$stride[3]

    # Residual path
    residual_out <- NULL
    if (self$residual) {
      # Reshape for pixel shuffle: [B, C, F, H, W] -> [B, C//s^3, s, s, s, F, H, W]
      residual_out <- hidden_states$reshape(c(
        batch_size,
        -1,
        self$stride[1], self$stride[2], self$stride[3],
        num_frames, height, width
      ))
      # Permute: [B, C', s_t, s_h, s_w, F, H, W] -> [B, C', F, s_t, H, s_h, W, s_w]
      residual_out <- residual_out$permute(c(1, 2, 6, 3, 7, 4, 8, 5))
      # Flatten spatial dims
      residual_out <- residual_out$flatten(start_dim = 7, end_dim = 8)  # W * s_w
      residual_out <- residual_out$flatten(start_dim = 5, end_dim = 6)  # H * s_h
      residual_out <- residual_out$flatten(start_dim = 3, end_dim = 4)  # F * s_t

      # Repeat channels for upscale factor
      repeats <- stride_prod %/% self$upscale_factor
      residual_out <- residual_out$`repeat`(c(1L, repeats, 1L, 1L, 1L))
      # Remove first (stride[1]-1) frames
      residual_out <- residual_out[, , self$stride[1]:N, , ]
    }

    # Convolution path
    hidden_states <- self$conv(hidden_states, causal = causal)

    # Pixel shuffle
    hidden_states <- hidden_states$reshape(c(
      batch_size,
      -1,
      self$stride[1], self$stride[2], self$stride[3],
      num_frames, height, width
    ))
    hidden_states <- hidden_states$permute(c(1, 2, 6, 3, 7, 4, 8, 5))
    hidden_states <- hidden_states$flatten(start_dim = 7, end_dim = 8)
    hidden_states <- hidden_states$flatten(start_dim = 5, end_dim = 6)
    hidden_states <- hidden_states$flatten(start_dim = 3, end_dim = 4)
    # Remove first (stride[1]-1) frames
    hidden_states <- hidden_states[, , self$stride[1]:N, , ]

    if (self$residual && !is.null(residual_out)) {
      hidden_states <- hidden_states + residual_out
    }

    hidden_states
  }
)

#' LTX2 Video Down Block 3D
#'
#' Encoder down block with multiple ResNet layers and optional downsampling.
#'
#' @param in_channels Integer. Input channels.
#' @param out_channels Integer or NULL. Output channels.
#' @param num_layers Integer. Number of ResNet layers.
#' @param dropout Numeric. Dropout rate.
#' @param resnet_eps Numeric. Epsilon for normalization.
#' @param resnet_act_fn Character. Activation function.
#' @param spatio_temporal_scale Logical. Whether to use downsampling.
#' @param downsample_type Character. Type of downsampling.
#' @param spatial_padding_mode Character. Padding mode.
#' @export
ltx2_video_down_block3d <- torch::nn_module(
  "LTX2VideoDownBlock3D",

  initialize = function(
    in_channels,
    out_channels = NULL,
    num_layers = 1L,
    dropout = 0.0,
    resnet_eps = 1e-6,
    resnet_act_fn = "swish",
    spatio_temporal_scale = TRUE,
    downsample_type = "conv",
    spatial_padding_mode = "zeros"
  ) {
    if (is.null(out_channels)) out_channels <- in_channels

    # ResNet layers
    resnets <- list()
    for (i in seq_len(num_layers)) {
      resnets[[i]] <- ltx2_video_resnet_block3d(
        in_channels = in_channels,
        out_channels = in_channels,  # Note: same channels within block
        dropout = dropout,
        eps = resnet_eps,
        non_linearity = resnet_act_fn,
        spatial_padding_mode = spatial_padding_mode
      )
    }
    self$resnets <- torch::nn_module_list(resnets)

    # Downsampler
    self$downsamplers <- NULL
    if (spatio_temporal_scale) {
      if (downsample_type == "conv") {
        self$downsamplers <- torch::nn_module_list(list(
          ltx2_video_causal_conv3d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = 3L,
            stride = c(2L, 2L, 2L),
            spatial_padding_mode = spatial_padding_mode
          )
        ))
      } else if (downsample_type == "spatial") {
        self$downsamplers <- torch::nn_module_list(list(
          ltx_video_downsampler3d(
            in_channels = in_channels,
            out_channels = out_channels,
            stride = c(1L, 2L, 2L),
            spatial_padding_mode = spatial_padding_mode
          )
        ))
      } else if (downsample_type == "temporal") {
        self$downsamplers <- torch::nn_module_list(list(
          ltx_video_downsampler3d(
            in_channels = in_channels,
            out_channels = out_channels,
            stride = c(2L, 1L, 1L),
            spatial_padding_mode = spatial_padding_mode
          )
        ))
      } else if (downsample_type == "spatiotemporal") {
        self$downsamplers <- torch::nn_module_list(list(
          ltx_video_downsampler3d(
            in_channels = in_channels,
            out_channels = out_channels,
            stride = c(2L, 2L, 2L),
            spatial_padding_mode = spatial_padding_mode
          )
        ))
      }
    }
  },

  forward = function(hidden_states, temb = NULL, generator = NULL, causal = TRUE) {
    for (i in seq_along(self$resnets)) {
      hidden_states <- self$resnets[[i]](hidden_states, temb, generator, causal = causal)
    }

    if (!is.null(self$downsamplers)) {
      for (i in seq_along(self$downsamplers)) {
        hidden_states <- self$downsamplers[[i]](hidden_states, causal = causal)
      }
    }

    hidden_states
  }
)

#' LTX2 Video Mid Block 3D
#'
#' Middle block with ResNet layers and optional timestep conditioning.
#'
#' @param in_channels Integer. Input channels.
#' @param num_layers Integer. Number of ResNet layers.
#' @param dropout Numeric. Dropout rate.
#' @param resnet_eps Numeric. Epsilon for normalization.
#' @param resnet_act_fn Character. Activation function.
#' @param inject_noise Logical. Whether to inject noise.
#' @param timestep_conditioning Logical. Whether to use timestep conditioning.
#' @param spatial_padding_mode Character. Padding mode.
#' @export
ltx2_video_mid_block3d <- torch::nn_module(
  "LTX2VideoMidBlock3d",

  initialize = function(
    in_channels,
    num_layers = 1L,
    dropout = 0.0,
    resnet_eps = 1e-6,
    resnet_act_fn = "swish",
    inject_noise = FALSE,
    timestep_conditioning = FALSE,
    spatial_padding_mode = "zeros"
  ) {
    # Time embedder for timestep conditioning (TODO: implement if needed)
    self$time_embedder <- NULL

    resnets <- list()
    for (i in seq_len(num_layers)) {
      resnets[[i]] <- ltx2_video_resnet_block3d(
        in_channels = in_channels,
        out_channels = in_channels,
        dropout = dropout,
        eps = resnet_eps,
        non_linearity = resnet_act_fn,
        inject_noise = inject_noise,
        timestep_conditioning = timestep_conditioning,
        spatial_padding_mode = spatial_padding_mode
      )
    }
    self$resnets <- torch::nn_module_list(resnets)
  },

  forward = function(hidden_states, temb = NULL, generator = NULL, causal = TRUE) {
    for (i in seq_along(self$resnets)) {
      hidden_states <- self$resnets[[i]](hidden_states, temb, generator, causal = causal)
    }
    hidden_states
  }
)

#' LTX2 Video Up Block 3D
#'
#' Decoder up block with upsampling and ResNet layers.
#'
#' @param in_channels Integer. Input channels.
#' @param out_channels Integer or NULL. Output channels.
#' @param num_layers Integer. Number of ResNet layers.
#' @param dropout Numeric. Dropout rate.
#' @param resnet_eps Numeric. Epsilon for normalization.
#' @param resnet_act_fn Character. Activation function.
#' @param spatio_temporal_scale Logical. Whether to use upsampling.
#' @param inject_noise Logical. Whether to inject noise.
#' @param timestep_conditioning Logical. Whether to use timestep conditioning.
#' @param upsample_residual Logical. Whether upsampler uses residual.
#' @param upscale_factor Integer. Channel upscale factor.
#' @param spatial_padding_mode Character. Padding mode.
#' @export
ltx2_video_up_block3d <- torch::nn_module(
  "LTX2VideoUpBlock3d",

  initialize = function(
    in_channels,
    out_channels = NULL,
    num_layers = 1L,
    dropout = 0.0,
    resnet_eps = 1e-6,
    resnet_act_fn = "swish",
    spatio_temporal_scale = TRUE,
    inject_noise = FALSE,
    timestep_conditioning = FALSE,
    upsample_residual = FALSE,
    upscale_factor = 1L,
    spatial_padding_mode = "zeros"
  ) {
    if (is.null(out_channels)) out_channels <- in_channels

    # Time embedder (TODO: implement if needed)
    self$time_embedder <- NULL

    # Input conv if channels differ
    self$conv_in <- NULL
    if (in_channels != out_channels) {
      self$conv_in <- ltx2_video_resnet_block3d(
        in_channels = in_channels,
        out_channels = out_channels,
        dropout = dropout,
        eps = resnet_eps,
        non_linearity = resnet_act_fn,
        inject_noise = inject_noise,
        timestep_conditioning = timestep_conditioning,
        spatial_padding_mode = spatial_padding_mode
      )
    }

    # Upsampler
    self$upsamplers <- NULL
    if (spatio_temporal_scale) {
      self$upsamplers <- torch::nn_module_list(list(
        ltx_video_upsampler3d(
          in_channels = out_channels * upscale_factor,
          stride = c(2L, 2L, 2L),
          residual = upsample_residual,
          upscale_factor = upscale_factor,
          spatial_padding_mode = spatial_padding_mode
        )
      ))
    }

    # ResNet layers
    resnets <- list()
    for (i in seq_len(num_layers)) {
      resnets[[i]] <- ltx2_video_resnet_block3d(
        in_channels = out_channels,
        out_channels = out_channels,
        dropout = dropout,
        eps = resnet_eps,
        non_linearity = resnet_act_fn,
        inject_noise = inject_noise,
        timestep_conditioning = timestep_conditioning,
        spatial_padding_mode = spatial_padding_mode
      )
    }
    self$resnets <- torch::nn_module_list(resnets)
  },

  forward = function(hidden_states, temb = NULL, generator = NULL, causal = TRUE) {
    if (!is.null(self$conv_in)) {
      hidden_states <- self$conv_in(hidden_states, temb, generator, causal = causal)
    }

    if (!is.null(self$upsamplers)) {
      for (i in seq_along(self$upsamplers)) {
        hidden_states <- self$upsamplers[[i]](hidden_states, causal = causal)
      }
    }

    for (i in seq_along(self$resnets)) {
      hidden_states <- self$resnets[[i]](hidden_states, temb, generator, causal = causal)
    }

    hidden_states
  }
)
