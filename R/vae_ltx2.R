#' LTX2 Video VAE
#'
#' 3D causal VAE for video encoding/decoding as used in LTX-2.
#' Supports GPU-poor tiling for memory-efficient processing.
#'
#' @name vae_ltx2
NULL

#' LTX2 Video Encoder
#'
#' Encodes video frames into latent space with 3D causal convolutions.
#'
#' @param in_channels Integer. Input channels (typically 3 for RGB).
#' @param out_channels Integer. Latent channels.
#' @param block_out_channels Integer vector. Output channels per block.
#' @param spatio_temporal_scaling Logical vector. Whether each block downscales.
#' @param layers_per_block Integer vector. Number of layers per block.
#' @param downsample_type Character vector. Type of downsampling per block.
#' @param patch_size Integer. Spatial patch size.
#' @param patch_size_t Integer. Temporal patch size.
#' @param resnet_norm_eps Numeric. Epsilon for normalization.
#' @param is_causal Logical. Whether to use causal convolutions.
#' @param spatial_padding_mode Character. Padding mode.
#' @export
ltx2_video_encoder3d <- torch::nn_module(
  "LTX2VideoEncoder3d",

  initialize = function(
    in_channels = 3L,
    out_channels = 128L,
    block_out_channels = c(256L, 512L, 1024L, 2048L),
    spatio_temporal_scaling = c(TRUE, TRUE, TRUE, TRUE),
    layers_per_block = c(4L, 6L, 6L, 2L, 2L),
    downsample_type = c("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
    patch_size = 4L,
    patch_size_t = 1L,
    resnet_norm_eps = 1e-6,
    is_causal = TRUE,
    spatial_padding_mode = "zeros"
  ) {
    self$patch_size <- patch_size
    self$patch_size_t <- patch_size_t
    self$in_channels <- in_channels * patch_size^2
    self$is_causal <- is_causal

    output_channel <- out_channels

    # Input convolution
    self$conv_in <- ltx2_video_causal_conv3d(
      in_channels = self$in_channels,
      out_channels = output_channel,
      kernel_size = 3L,
      stride = 1L,
      spatial_padding_mode = spatial_padding_mode
    )

    # Down blocks
    num_blocks <- length(block_out_channels)
    down_blocks <- list()
    for (i in seq_len(num_blocks)) {
      input_channel <- output_channel
      output_channel <- block_out_channels[i]

      down_blocks[[i]] <- ltx2_video_down_block3d(
        in_channels = input_channel,
        out_channels = output_channel,
        num_layers = layers_per_block[i],
        resnet_eps = resnet_norm_eps,
        spatio_temporal_scale = spatio_temporal_scaling[i],
        downsample_type = downsample_type[i],
        spatial_padding_mode = spatial_padding_mode
      )
    }
    self$down_blocks <- torch::nn_module_list(down_blocks)

    # Mid block
    self$mid_block <- ltx2_video_mid_block3d(
      in_channels = output_channel,
      num_layers = layers_per_block[length(layers_per_block)],
      resnet_eps = resnet_norm_eps,
      spatial_padding_mode = spatial_padding_mode
    )

    # Output
    self$norm_out <- per_channel_rms_norm()
    self$conv_act <- torch::nn_silu()
    self$conv_out <- ltx2_video_causal_conv3d(
      in_channels = output_channel,
      out_channels = out_channels + 1L,  # +1 for log variance
      kernel_size = 3L,
      stride = 1L,
      spatial_padding_mode = spatial_padding_mode
    )
  },

  forward = function(hidden_states, causal = NULL) {
    p <- self$patch_size
    p_t <- self$patch_size_t

    batch_size <- hidden_states$shape[1]
    num_channels <- hidden_states$shape[2]
    num_frames <- hidden_states$shape[3]
    height <- hidden_states$shape[4]
    width <- hidden_states$shape[5]

    post_patch_num_frames <- num_frames %/% p_t
    post_patch_height <- height %/% p
    post_patch_width <- width %/% p

    if (is.null(causal)) causal <- self$is_causal

    # Patchify: reshape to separate patches
    hidden_states <- hidden_states$reshape(c(
      batch_size, num_channels,
      post_patch_num_frames, p_t,
      post_patch_height, p,
      post_patch_width, p
    ))
    # Permute to channel-first for patches
    # [B, C, F', p_t, H', p, W', p] -> [B, C, p_t, p, p, F', H', W']
    hidden_states <- hidden_states$permute(c(1, 2, 4, 8, 6, 3, 5, 7))
    hidden_states <- hidden_states$flatten(start_dim = 2, end_dim = 5)

    hidden_states <- self$conv_in(hidden_states, causal = causal)

    for (i in seq_along(self$down_blocks)) {
      hidden_states <- self$down_blocks[[i]](hidden_states, causal = causal)
    }

    hidden_states <- self$mid_block(hidden_states, causal = causal)

    hidden_states <- self$norm_out(hidden_states)
    hidden_states <- self$conv_act(hidden_states)
    hidden_states <- self$conv_out(hidden_states, causal = causal)

    # Duplicate last channel for mean/logvar split
    last_channel <- hidden_states[, -1, , , ]$unsqueeze(2)
    last_channel <- last_channel$`repeat`(c(1L, hidden_states$shape[2] - 2L, 1L, 1L, 1L))
    hidden_states <- torch::torch_cat(list(hidden_states, last_channel), dim = 2)

    hidden_states
  }
)

#' LTX2 Video Decoder
#'
#' Decodes latent representations back to video frames.
#'
#' @param in_channels Integer. Latent channels.
#' @param out_channels Integer. Output channels (typically 3 for RGB).
#' @param block_out_channels Integer vector. Output channels per block.
#' @param spatio_temporal_scaling Logical vector. Whether each block upscales.
#' @param layers_per_block Integer vector. Number of layers per block.
#' @param patch_size Integer. Spatial patch size.
#' @param patch_size_t Integer. Temporal patch size.
#' @param resnet_norm_eps Numeric. Epsilon for normalization.
#' @param is_causal Logical. Whether to use causal convolutions.
#' @param inject_noise Logical vector. Whether to inject noise per block.
#' @param timestep_conditioning Logical. Whether to use timestep conditioning.
#' @param upsample_residual Logical vector. Whether upsamplers use residual.
#' @param upsample_factor Integer vector. Channel upscale factors.
#' @param spatial_padding_mode Character. Padding mode.
#' @export
ltx2_video_decoder3d <- torch::nn_module(
  "LTX2VideoDecoder3d",

  initialize = function(
    in_channels = 128L,
    out_channels = 3L,
    block_out_channels = c(256L, 512L, 1024L),
    spatio_temporal_scaling = c(TRUE, TRUE, TRUE),
    layers_per_block = c(5L, 5L, 5L, 5L),
    patch_size = 4L,
    patch_size_t = 1L,
    resnet_norm_eps = 1e-6,
    is_causal = FALSE,
    inject_noise = c(FALSE, FALSE, FALSE, FALSE),
    timestep_conditioning = FALSE,
    upsample_residual = c(TRUE, TRUE, TRUE),
    upsample_factor = c(2L, 2L, 2L),
    spatial_padding_mode = "reflect"
  ) {
    self$patch_size <- patch_size
    self$patch_size_t <- patch_size_t
    self$out_channels <- out_channels * patch_size^2
    self$is_causal <- is_causal

    # Reverse orders for decoder
    block_out_channels <- rev(block_out_channels)
    spatio_temporal_scaling <- rev(spatio_temporal_scaling)
    layers_per_block <- rev(layers_per_block)
    inject_noise <- rev(inject_noise)
    upsample_residual <- rev(upsample_residual)
    upsample_factor <- rev(upsample_factor)

    output_channel <- block_out_channels[1]

    # Input convolution
    self$conv_in <- ltx2_video_causal_conv3d(
      in_channels = in_channels,
      out_channels = output_channel,
      kernel_size = 3L,
      stride = 1L,
      spatial_padding_mode = spatial_padding_mode
    )

    # Mid block
    self$mid_block <- ltx2_video_mid_block3d(
      in_channels = output_channel,
      num_layers = layers_per_block[1],
      resnet_eps = resnet_norm_eps,
      inject_noise = inject_noise[1],
      timestep_conditioning = timestep_conditioning,
      spatial_padding_mode = spatial_padding_mode
    )

    # Up blocks
    num_blocks <- length(block_out_channels)
    up_blocks <- list()
    for (i in seq_len(num_blocks)) {
      input_channel <- output_channel %/% upsample_factor[i]
      output_channel <- block_out_channels[i] %/% upsample_factor[i]

      up_blocks[[i]] <- ltx2_video_up_block3d(
        in_channels = input_channel,
        out_channels = output_channel,
        num_layers = layers_per_block[i + 1],
        resnet_eps = resnet_norm_eps,
        spatio_temporal_scale = spatio_temporal_scaling[i],
        inject_noise = if (i + 1 <= length(inject_noise)) inject_noise[i + 1] else FALSE,
        timestep_conditioning = timestep_conditioning,
        upsample_residual = upsample_residual[i],
        upscale_factor = upsample_factor[i],
        spatial_padding_mode = spatial_padding_mode
      )
    }
    self$up_blocks <- torch::nn_module_list(up_blocks)

    # Output
    self$norm_out <- per_channel_rms_norm()
    self$conv_act <- torch::nn_silu()
    self$conv_out <- ltx2_video_causal_conv3d(
      in_channels = output_channel,
      out_channels = self$out_channels,
      kernel_size = 3L,
      stride = 1L,
      spatial_padding_mode = spatial_padding_mode
    )

    # Timestep embedding (optional)
    self$time_embedder <- NULL
    self$scale_shift_table <- NULL
    self$timestep_scale_multiplier <- NULL
  },

  forward = function(hidden_states, temb = NULL, causal = NULL) {
    if (is.null(causal)) causal <- self$is_causal

    hidden_states <- self$conv_in(hidden_states, causal = causal)

    if (!is.null(self$timestep_scale_multiplier) && !is.null(temb)) {
      temb <- temb * self$timestep_scale_multiplier
    }

    hidden_states <- self$mid_block(hidden_states, temb, causal = causal)

    for (i in seq_along(self$up_blocks)) {
      hidden_states <- self$up_blocks[[i]](hidden_states, temb, causal = causal)
    }

    hidden_states <- self$norm_out(hidden_states)
    hidden_states <- self$conv_act(hidden_states)
    hidden_states <- self$conv_out(hidden_states, causal = causal)

    # Unpatchify: reshape to original spatial dims
    p <- self$patch_size
    p_t <- self$patch_size_t

    batch_size <- hidden_states$shape[1]
    num_channels <- hidden_states$shape[2]
    num_frames <- hidden_states$shape[3]
    height <- hidden_states$shape[4]
    width <- hidden_states$shape[5]

    # [B, C*p_t*p*p, F, H, W] -> [B, C, p_t, p, p, F, H, W]
    hidden_states <- hidden_states$reshape(c(
      batch_size, -1, p_t, p, p, num_frames, height, width
    ))
    # Permute: [B, C, p_t, p, p, F, H, W] -> [B, C, F, p_t, H, p, W, p]
    hidden_states <- hidden_states$permute(c(1, 2, 6, 3, 7, 5, 8, 4))
    # Flatten to full resolution
    hidden_states <- hidden_states$flatten(start_dim = 7, end_dim = 8)  # W*p
    hidden_states <- hidden_states$flatten(start_dim = 5, end_dim = 6)  # H*p
    hidden_states <- hidden_states$flatten(start_dim = 3, end_dim = 4)  # F*p_t

    hidden_states
  }
)

#' Diagonal Gaussian Distribution
#'
#' Represents a diagonal Gaussian distribution for VAE latents.
#'
#' @param parameters Tensor of concatenated mean and log variance.
#' @export
diagonal_gaussian_distribution <- function(parameters) {
  # Split parameters into mean and logvar
  chunk_dim <- 2  # Channel dimension
  mean_logvar <- parameters$chunk(2, dim = chunk_dim)
  mean <- mean_logvar[[1]]
  logvar <- mean_logvar[[2]]

  # Clamp logvar for numerical stability
  logvar <- logvar$clamp(min = -30.0, max = 20.0)
  std <- torch::torch_exp(0.5 * logvar)
  var <- torch::torch_exp(logvar)

  list(
    mean = mean,
    logvar = logvar,
    std = std,
    var = var,
    sample = function(generator = NULL) {
      sample <- torch::torch_randn_like(mean)
      mean + std * sample
    },
    mode = function() mean
  )
}

#' LTX2 Video VAE
#'
#' Full VAE with encoder and decoder, supporting tiled encoding/decoding
#' for GPU-poor memory management.
#'
#' @param in_channels Integer. Input channels.
#' @param out_channels Integer. Output channels.
#' @param latent_channels Integer. Latent space channels.
#' @param block_out_channels Integer vector. Encoder block channels.
#' @param decoder_block_out_channels Integer vector. Decoder block channels.
#' @param layers_per_block Integer vector. Encoder layers per block.
#' @param decoder_layers_per_block Integer vector. Decoder layers per block.
#' @param spatio_temporal_scaling Logical vector. Encoder scaling.
#' @param decoder_spatio_temporal_scaling Logical vector. Decoder scaling.
#' @param decoder_inject_noise Logical vector. Noise injection per decoder block.
#' @param downsample_type Character vector. Downsampling types.
#' @param upsample_residual Logical vector. Upsampler residual flags.
#' @param upsample_factor Integer vector. Upsampler factors.
#' @param timestep_conditioning Logical. Whether to use timestep conditioning.
#' @param patch_size Integer. Spatial patch size.
#' @param patch_size_t Integer. Temporal patch size.
#' @param resnet_norm_eps Numeric. Normalization epsilon.
#' @param scaling_factor Numeric. Latent scaling factor.
#' @param encoder_causal Logical. Encoder causality.
#' @param decoder_causal Logical. Decoder causality.
#' @param encoder_spatial_padding_mode Character. Encoder padding mode.
#' @param decoder_spatial_padding_mode Character. Decoder padding mode.
#' @export
ltx2_video_vae <- torch::nn_module(
  "AutoencoderKLLTX2Video",

  initialize = function(
    in_channels = 3L,
    out_channels = 3L,
    latent_channels = 128L,
    block_out_channels = c(256L, 512L, 1024L, 2048L),
    decoder_block_out_channels = c(256L, 512L, 1024L),
    layers_per_block = c(4L, 6L, 6L, 2L, 2L),
    decoder_layers_per_block = c(5L, 5L, 5L, 5L),
    spatio_temporal_scaling = c(TRUE, TRUE, TRUE, TRUE),
    decoder_spatio_temporal_scaling = c(TRUE, TRUE, TRUE),
    decoder_inject_noise = c(FALSE, FALSE, FALSE, FALSE),
    downsample_type = c("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
    upsample_residual = c(TRUE, TRUE, TRUE),
    upsample_factor = c(2L, 2L, 2L),
    timestep_conditioning = FALSE,
    patch_size = 4L,
    patch_size_t = 1L,
    resnet_norm_eps = 1e-6,
    scaling_factor = 1.0,
    encoder_causal = TRUE,
    decoder_causal = TRUE,
    encoder_spatial_padding_mode = "zeros",
    decoder_spatial_padding_mode = "reflect"
  ) {
    self$encoder <- ltx2_video_encoder3d(
      in_channels = in_channels,
      out_channels = latent_channels,
      block_out_channels = block_out_channels,
      spatio_temporal_scaling = spatio_temporal_scaling,
      layers_per_block = layers_per_block,
      downsample_type = downsample_type,
      patch_size = patch_size,
      patch_size_t = patch_size_t,
      resnet_norm_eps = resnet_norm_eps,
      is_causal = encoder_causal,
      spatial_padding_mode = encoder_spatial_padding_mode
    )

    self$decoder <- ltx2_video_decoder3d(
      in_channels = latent_channels,
      out_channels = out_channels,
      block_out_channels = decoder_block_out_channels,
      spatio_temporal_scaling = decoder_spatio_temporal_scaling,
      layers_per_block = decoder_layers_per_block,
      patch_size = patch_size,
      patch_size_t = patch_size_t,
      resnet_norm_eps = resnet_norm_eps,
      is_causal = decoder_causal,
      inject_noise = decoder_inject_noise,
      timestep_conditioning = timestep_conditioning,
      upsample_residual = upsample_residual,
      upsample_factor = upsample_factor,
      spatial_padding_mode = decoder_spatial_padding_mode
    )

    # Latent normalization buffers
    self$latents_mean <- torch::nn_buffer(torch::torch_zeros(latent_channels))
    self$latents_std <- torch::nn_buffer(torch::torch_ones(latent_channels))

    # Compression ratios
    self$spatial_compression_ratio <- patch_size * 2^sum(spatio_temporal_scaling)
    self$temporal_compression_ratio <- patch_size_t * 2^sum(spatio_temporal_scaling)

    self$scaling_factor <- scaling_factor

    # Tiling configuration (GPU-poor support)
    self$use_slicing <- FALSE
    self$use_tiling <- FALSE
    self$use_framewise_encoding <- FALSE
    self$use_framewise_decoding <- FALSE

    self$num_sample_frames_batch_size <- 16L
    self$num_latent_frames_batch_size <- 2L

    self$tile_sample_min_height <- 512L
    self$tile_sample_min_width <- 512L
    self$tile_sample_min_num_frames <- 16L

    self$tile_sample_stride_height <- 448L
    self$tile_sample_stride_width <- 448L
    self$tile_sample_stride_num_frames <- 8L
  },

  #' Enable tiled encoding/decoding for memory efficiency
  enable_tiling = function(
    tile_sample_min_height = NULL,
    tile_sample_min_width = NULL,
    tile_sample_min_num_frames = NULL,
    tile_sample_stride_height = NULL,
    tile_sample_stride_width = NULL,
    tile_sample_stride_num_frames = NULL
  ) {
    self$use_tiling <- TRUE
    if (!is.null(tile_sample_min_height)) self$tile_sample_min_height <- tile_sample_min_height
    if (!is.null(tile_sample_min_width)) self$tile_sample_min_width <- tile_sample_min_width
    if (!is.null(tile_sample_min_num_frames)) self$tile_sample_min_num_frames <- tile_sample_min_num_frames
    if (!is.null(tile_sample_stride_height)) self$tile_sample_stride_height <- tile_sample_stride_height
    if (!is.null(tile_sample_stride_width)) self$tile_sample_stride_width <- tile_sample_stride_width
    if (!is.null(tile_sample_stride_num_frames)) self$tile_sample_stride_num_frames <- tile_sample_stride_num_frames
    invisible(self)
  },

  #' Disable tiling
  disable_tiling = function() {
    self$use_tiling <- FALSE
    invisible(self)
  },

  #' Enable framewise decoding for long videos
  enable_framewise_decoding = function() {
    self$use_framewise_decoding <- TRUE
    invisible(self)
  },

  #' Blend tiles vertically
  blend_v = function(a, b, blend_extent) {
    blend_extent <- min(a$shape[4], b$shape[4], blend_extent)
    for (y in seq_len(blend_extent)) {
      weight <- (y - 1) / blend_extent
      idx <- as.integer(a$shape[4] - blend_extent + y)
      b[, , , y, ] <- a[, , , idx, ] * (1 - weight) + b[, , , y, ] * weight
    }
    b
  },

  #' Blend tiles horizontally
  blend_h = function(a, b, blend_extent) {
    blend_extent <- min(a$shape[5], b$shape[5], blend_extent)
    for (x in seq_len(blend_extent)) {
      weight <- (x - 1) / blend_extent
      idx <- as.integer(a$shape[5] - blend_extent + x)
      b[, , , , x] <- a[, , , , idx] * (1 - weight) + b[, , , , x] * weight
    }
    b
  },

  #' Blend tiles temporally
  blend_t = function(a, b, blend_extent) {
    blend_extent <- min(a$shape[3], b$shape[3], blend_extent)
    for (t in seq_len(blend_extent)) {
      weight <- (t - 1) / blend_extent
      idx <- as.integer(a$shape[3] - blend_extent + t)
      b[, , t, , ] <- a[, , idx, , ] * (1 - weight) + b[, , t, , ] * weight
    }
    b
  },

  #' Tiled encoding for large spatial dimensions
  tiled_encode = function(x, causal = NULL) {
    batch_size <- x$shape[1]
    num_channels <- x$shape[2]
    num_frames <- x$shape[3]
    height <- x$shape[4]
    width <- x$shape[5]

    latent_height <- height %/% self$spatial_compression_ratio
    latent_width <- width %/% self$spatial_compression_ratio

    tile_latent_min_height <- self$tile_sample_min_height %/% self$spatial_compression_ratio
    tile_latent_min_width <- self$tile_sample_min_width %/% self$spatial_compression_ratio
    tile_latent_stride_height <- self$tile_sample_stride_height %/% self$spatial_compression_ratio
    tile_latent_stride_width <- self$tile_sample_stride_width %/% self$spatial_compression_ratio

    blend_height <- tile_latent_min_height - tile_latent_stride_height
    blend_width <- tile_latent_min_width - tile_latent_stride_width

    # Encode tiles
    rows <- list()
    for (i in seq(1, height, by = self$tile_sample_stride_height)) {
      row <- list()
      for (j in seq(1, width, by = self$tile_sample_stride_width)) {
        i_end <- min(i + self$tile_sample_min_height - 1, height)
        j_end <- min(j + self$tile_sample_min_width - 1, width)
        tile <- self$encoder(x[, , , i:i_end, j:j_end], causal = causal)
        row[[length(row) + 1]] <- tile
      }
      rows[[length(rows) + 1]] <- row
    }

    # Blend tiles
    result_rows <- list()
    for (i in seq_along(rows)) {
      result_row <- list()
      for (j in seq_along(rows[[i]])) {
        tile <- rows[[i]][[j]]
        if (i > 1) {
          tile <- self$blend_v(rows[[i - 1]][[j]], tile, blend_height)
        }
        if (j > 1) {
          tile <- self$blend_h(rows[[i]][[j - 1]], tile, blend_width)
        }
        result_row[[length(result_row) + 1]] <- tile[, , , 1:tile_latent_stride_height, 1:tile_latent_stride_width]
      }
      result_rows[[length(result_rows) + 1]] <- torch::torch_cat(result_row, dim = 5)
    }

    enc <- torch::torch_cat(result_rows, dim = 4)[, , , 1:latent_height, 1:latent_width]
    enc
  },

  #' Tiled decoding for large spatial dimensions
  tiled_decode = function(z, temb = NULL, causal = NULL) {
    batch_size <- z$shape[1]
    num_channels <- z$shape[2]
    num_frames <- z$shape[3]
    height <- z$shape[4]
    width <- z$shape[5]

    sample_height <- height * self$spatial_compression_ratio
    sample_width <- width * self$spatial_compression_ratio

    tile_latent_min_height <- self$tile_sample_min_height %/% self$spatial_compression_ratio
    tile_latent_min_width <- self$tile_sample_min_width %/% self$spatial_compression_ratio
    tile_latent_stride_height <- self$tile_sample_stride_height %/% self$spatial_compression_ratio
    tile_latent_stride_width <- self$tile_sample_stride_width %/% self$spatial_compression_ratio

    blend_height <- self$tile_sample_min_height - self$tile_sample_stride_height
    blend_width <- self$tile_sample_min_width - self$tile_sample_stride_width

    # Decode tiles
    rows <- list()
    for (i in seq(1, height, by = tile_latent_stride_height)) {
      row <- list()
      for (j in seq(1, width, by = tile_latent_stride_width)) {
        i_end <- min(i + tile_latent_min_height - 1, height)
        j_end <- min(j + tile_latent_min_width - 1, width)
        tile <- self$decoder(z[, , , i:i_end, j:j_end], temb, causal = causal)
        row[[length(row) + 1]] <- tile
      }
      rows[[length(rows) + 1]] <- row
    }

    # Blend tiles
    result_rows <- list()
    for (i in seq_along(rows)) {
      result_row <- list()
      for (j in seq_along(rows[[i]])) {
        tile <- rows[[i]][[j]]
        if (i > 1) {
          tile <- self$blend_v(rows[[i - 1]][[j]], tile, blend_height)
        }
        if (j > 1) {
          tile <- self$blend_h(rows[[i]][[j - 1]], tile, blend_width)
        }
        result_row[[length(result_row) + 1]] <- tile[, , , 1:self$tile_sample_stride_height, 1:self$tile_sample_stride_width]
      }
      result_rows[[length(result_rows) + 1]] <- torch::torch_cat(result_row, dim = 5)
    }

    dec <- torch::torch_cat(result_rows, dim = 4)[, , , 1:sample_height, 1:sample_width]
    dec
  },

  #' Internal encode with tiling support
  .encode = function(x, causal = NULL) {
    batch_size <- x$shape[1]
    num_channels <- x$shape[2]
    num_frames <- x$shape[3]
    height <- x$shape[4]
    width <- x$shape[5]

    if (self$use_tiling && (width > self$tile_sample_min_width || height > self$tile_sample_min_height)) {
      return(self$tiled_encode(x, causal = causal))
    }

    self$encoder(x, causal = causal)
  },

  #' Encode video to latent space
  encode = function(x, causal = NULL) {
    if (self$use_slicing && x$shape[1] > 1) {
      encoded_slices <- lapply(seq_len(x$shape[1]), function(i) {
        self$.encode(x[i:i, , , , , drop = FALSE], causal = causal)
      })
      h <- torch::torch_cat(encoded_slices, dim = 1)
    } else {
      h <- self$.encode(x, causal = causal)
    }
    diagonal_gaussian_distribution(h)
  },

  #' Internal decode with tiling support
  .decode = function(z, temb = NULL, causal = NULL) {
    batch_size <- z$shape[1]
    num_channels <- z$shape[2]
    num_frames <- z$shape[3]
    height <- z$shape[4]
    width <- z$shape[5]

    tile_latent_min_height <- self$tile_sample_min_height %/% self$spatial_compression_ratio
    tile_latent_min_width <- self$tile_sample_min_width %/% self$spatial_compression_ratio

    if (self$use_tiling && (width > tile_latent_min_width || height > tile_latent_min_height)) {
      return(self$tiled_decode(z, temb, causal = causal))
    }

    self$decoder(z, temb, causal = causal)
  },

  #' Decode latent to video
  decode = function(z, temb = NULL, causal = NULL) {
    if (self$use_slicing && z$shape[1] > 1) {
      if (!is.null(temb)) {
        decoded_slices <- lapply(seq_len(z$shape[1]), function(i) {
          self$.decode(z[i:i, , , , , drop = FALSE], temb[i:i, drop = FALSE], causal = causal)
        })
      } else {
        decoded_slices <- lapply(seq_len(z$shape[1]), function(i) {
          self$.decode(z[i:i, , , , , drop = FALSE], causal = causal)
        })
      }
      torch::torch_cat(decoded_slices, dim = 1)
    } else {
      self$.decode(z, temb, causal = causal)
    }
  },

  #' Full forward pass (encode -> sample/mode -> decode)
  forward = function(
    sample,
    temb = NULL,
    sample_posterior = FALSE,
    encoder_causal = NULL,
    decoder_causal = NULL,
    generator = NULL
  ) {
    posterior <- self$encode(sample, causal = encoder_causal)
    if (sample_posterior) {
      z <- posterior$sample(generator)
    } else {
      z <- posterior$mode()
    }
    self$decode(z, temb, causal = decoder_causal)
  }
)

#' Load LTX2 Video VAE from safetensors
#'
#' Load pre-trained LTX2 VAE weights from a HuggingFace safetensors file.
#'
#' @param weights_path Character. Path to safetensors file or directory containing weights.
#' @param config_path Character. Optional path to config.json. If NULL and weights_path
#'   is a directory, looks for config.json in that directory. Otherwise uses default config.
#' @param device Character. Device to load weights to. Default: "cpu"
#' @param dtype Character or torch dtype. Data type. Default: "float32"
#' @param verbose Logical. Print loading progress. Default: TRUE
#' @return Initialized ltx2_video_vae module
#' @export
load_ltx2_vae <- function(weights_path, config_path = NULL, device = "cpu",
                          dtype = "float32", verbose = TRUE) {
  if (!file.exists(weights_path)) {
    stop("Weights path not found: ", weights_path)
  }

  # Handle directory vs file
  if (dir.exists(weights_path)) {
    vae_dir <- weights_path
    # Look for config.json
    if (is.null(config_path)) {
      config_path <- file.path(vae_dir, "config.json")
      if (!file.exists(config_path)) config_path <- NULL
    }
    # Look for weights file
    weights_file <- file.path(vae_dir, "diffusion_pytorch_model.safetensors")
    if (!file.exists(weights_file)) {
      stop("Could not find diffusion_pytorch_model.safetensors in: ", vae_dir)
    }
    weights_path <- weights_file
  }

  # Load config if provided
  config <- NULL
  if (!is.null(config_path) && file.exists(config_path)) {
    config <- jsonlite::fromJSON(config_path)
    if (verbose) message("Loaded config from: ", config_path)
  }

  # Create VAE with config or defaults (matching HuggingFace LTX-2)
  if (!is.null(config)) {
    vae <- ltx2_video_vae(
      in_channels = config$in_channels %||% 3L,
      out_channels = config$out_channels %||% 3L,
      latent_channels = config$latent_channels %||% 128L,
      block_out_channels = as.integer(config$block_out_channels %||% c(256, 512, 1024, 2048)),
      decoder_block_out_channels = as.integer(config$decoder_block_out_channels %||% c(256, 512, 1024)),
      layers_per_block = as.integer(config$layers_per_block %||% c(4, 6, 6, 2, 2)),
      decoder_layers_per_block = as.integer(config$decoder_layers_per_block %||% c(5, 5, 5, 5)),
      spatio_temporal_scaling = as.logical(config$spatio_temporal_scaling %||% c(TRUE, TRUE, TRUE, TRUE)),
      decoder_spatio_temporal_scaling = as.logical(config$decoder_spatio_temporal_scaling %||% c(TRUE, TRUE, TRUE)),
      decoder_inject_noise = as.logical(config$decoder_inject_noise %||% c(FALSE, FALSE, FALSE, FALSE)),
      downsample_type = config$downsample_type %||% c("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
      upsample_residual = as.logical(config$upsample_residual %||% c(TRUE, TRUE, TRUE)),
      upsample_factor = as.integer(config$upsample_factor %||% c(2, 2, 2)),
      timestep_conditioning = config$timestep_conditioning %||% FALSE,
      patch_size = config$patch_size %||% 4L,
      patch_size_t = config$patch_size_t %||% 1L,
      resnet_norm_eps = config$resnet_norm_eps %||% 1e-6,
      scaling_factor = config$scaling_factor %||% 1.0,
      encoder_causal = config$encoder_causal %||% TRUE,
      decoder_causal = config$decoder_causal %||% FALSE,
      encoder_spatial_padding_mode = config$encoder_spatial_padding_mode %||% "zeros",
      decoder_spatial_padding_mode = config$decoder_spatial_padding_mode %||% "reflect"
    )
  } else {
    vae <- ltx2_video_vae()
  }

  # Load weights
  if (verbose) message("Loading weights from: ", weights_path)
  weights <- safetensors::safe_load_file(weights_path, framework = "torch")

  # Load weights into VAE
  load_ltx2_vae_weights(vae, weights, verbose = verbose)

  # Move to device with dtype
  torch_dtype <- if (dtype == "float16") torch::torch_float16() else torch::torch_float32()
  vae$to(device = device, dtype = torch_dtype)

  if (verbose) message("VAE loaded successfully on device: ", device)
  vae
}

#' Load weights into LTX2 VAE module
#'
#' Maps HuggingFace safetensors parameter names to R module parameters.
#'
#' @param vae LTX2 VAE module
#' @param weights Named list of weight tensors from safetensors
#' @param verbose Logical. Print loading progress
#' @return The VAE with loaded weights (invisibly)
#' @keywords internal
load_ltx2_vae_weights <- function(vae, weights, verbose = TRUE) {
  # Get native parameter names
  native_params <- names(vae$parameters)

  # Build mapping from HF names to R names
  remap_vae_key <- function(key) {
    # HuggingFace VAE naming:
    # encoder.conv_in.conv.weight -> encoder.conv_in.conv.weight
    # encoder.down_blocks.0.resnets.0.norm.weight -> encoder.down_blocks.0.resnets.0.norm.weight
    # The naming should be mostly 1:1 with our R module structure

    # No remapping needed for most keys - HF uses same structure
    key
  }

  loaded <- 0L
  skipped <- 0L
  unmapped <- character(0)

  torch::with_no_grad({
    for (hf_name in names(weights)) {
      native_name <- remap_vae_key(hf_name)

      if (native_name %in% native_params) {
        hf_tensor <- weights[[hf_name]]
        native_tensor <- vae$parameters[[native_name]]

        if (all(as.integer(hf_tensor$shape) == as.integer(native_tensor$shape))) {
          native_tensor$copy_(hf_tensor)
          loaded <- loaded + 1L
        } else {
          if (verbose) {
            message("Shape mismatch: ", native_name,
                    " (HF: ", paste(as.integer(hf_tensor$shape), collapse = "x"),
                    " vs R: ", paste(as.integer(native_tensor$shape), collapse = "x"), ")")
          }
          skipped <- skipped + 1L
        }
      } else {
        skipped <- skipped + 1L
        unmapped <- c(unmapped, paste0(hf_name, " -> ", native_name))
      }
    }
  })

  if (verbose) {
    message(sprintf("VAE weights: %d loaded, %d skipped", loaded, skipped))
    if (length(unmapped) > 0 && length(unmapped) <= 20) {
      message("Unmapped parameters:")
      for (u in unmapped[1:min(20, length(unmapped))]) {
        message("  ", u)
      }
    }
    if (length(unmapped) > 20) {
      message("  ... and ", length(unmapped) - 20, " more")
    }
  }

  invisible(vae)
}

#' Null-coalescing operator
#' @keywords internal
`%||%` <- function(x, y) if (is.null(x)) y else x
