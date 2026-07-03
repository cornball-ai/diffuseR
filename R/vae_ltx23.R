#' LTX-2.3 Causal Video VAE
#'
#' Fresh R port of the LTX-2 video autoencoder from the diffusers
#' reference (Apache-2.0, autoencoder_kl_ltx2.py), with LTX 2.3 defaults:
#' encoder blocks (256, 512, 1024, 1024), a 4-up-block decoder with mixed
#' (spatiotemporal, spatiotemporal, temporal, spatial) upsampling, no
#' upsample residuals, and zeros spatial padding throughout. The encoder
#' is causal; the decoder is not.
#'
#' @name vae_ltx23
NULL

#' LTX-2.3 video encoder
#'
#' Pixel video [B, 3, F, H, W] -> latent statistics
#' [B, 2 * latent_channels, F/8, H/32, W/32] (mean and a uniform log-var
#' channel broadcast across the latent channels).
#'
#' @param in_channels,out_channels Integers. Pixel and latent channels.
#' @param block_out_channels Integer vector. Per-block output channels.
#' @param spatio_temporal_scaling Logical vector per block.
#' @param layers_per_block Integer vector (blocks then mid).
#' @param downsample_type Character vector per block.
#' @param patch_size,patch_size_t Integers. Pixel patchification.
#' @param resnet_norm_eps Numeric.
#' @param is_causal Logical.
#' @param spatial_padding_mode Character.
#'
#' @export
ltx23_video_encoder3d <- torch::nn_module(
  "ltx23_video_encoder3d",
  initialize = function(
    in_channels = 3L,
    out_channels = 128L,
    block_out_channels = c(256L, 512L, 1024L, 1024L),
    spatio_temporal_scaling = c(TRUE, TRUE, TRUE, TRUE),
    layers_per_block = c(4L, 6L, 4L, 2L, 2L),
    downsample_type = c("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
    patch_size = 4L,
    patch_size_t = 1L,
    resnet_norm_eps = 1e-6,
    is_causal = TRUE,
    spatial_padding_mode = "zeros"
  ) {
    self$patch_size <- as.integer(patch_size)
    self$patch_size_t <- as.integer(patch_size_t)
    self$in_channels <- in_channels * patch_size^2
    self$is_causal <- is_causal

    output_channel <- out_channels
    self$conv_in <- ltx23_causal_conv3d(
      self$in_channels, output_channel, kernel_size = 3L, stride = 1L,
      spatial_padding_mode = spatial_padding_mode
    )

    down_blocks <- list()
    for (i in seq_along(block_out_channels)) {
      input_channel <- output_channel
      output_channel <- block_out_channels[i]
      down_blocks[[i]] <- ltx23_video_down_block3d(
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

    self$mid_block <- ltx23_video_mid_block3d(
      in_channels = output_channel,
      num_layers = layers_per_block[length(layers_per_block)],
      resnet_eps = resnet_norm_eps,
      spatial_padding_mode = spatial_padding_mode
    )

    self$norm_out <- ltx23_per_channel_rms_norm()
    self$conv_out <- ltx23_causal_conv3d(
      output_channel, out_channels + 1L, kernel_size = 3L, stride = 1L,
      spatial_padding_mode = spatial_padding_mode
    )
  },
  forward = function(hidden_states, causal = NULL) {
    causal <- causal %||% self$is_causal
    p <- self$patch_size
    p_t <- self$patch_size_t

    dims <- hidden_states$shape
    batch_size <- dims[1]; num_channels <- dims[2]
    post_f <- dims[3] %/% p_t; post_h <- dims[4] %/% p; post_w <- dims[5] %/% p

    # Pixel patchification: space-to-channel with the LTX ordering
    hidden_states <- hidden_states$reshape(c(
      batch_size, num_channels, post_f, p_t, post_h, p, post_w, p
    ))
    # [B, C, F', pt, H', p, W', p] -> [B, C, pt, pw, ph, F', H', W']
    hidden_states <- hidden_states$permute(c(1L, 2L, 4L, 8L, 6L, 3L, 5L, 7L))$
      flatten(start_dim = 2L, end_dim = 5L)
    hidden_states <- self$conv_in(hidden_states, causal = causal)

    for (i in seq_along(self$down_blocks)) {
      hidden_states <- self$down_blocks[[i]](hidden_states, causal = causal)
    }
    hidden_states <- self$mid_block(hidden_states, causal = causal)

    hidden_states <- self$norm_out(hidden_states)
    hidden_states <- torch::nnf_silu(hidden_states)
    hidden_states <- self$conv_out(hidden_states, causal = causal)

    # Broadcast the single log-var channel across all latent channels
    n_ch <- hidden_states$shape[2]
    last_channel <- hidden_states$narrow(2L, n_ch, 1L)$`repeat`(c(1L, n_ch - 2L, 1L, 1L, 1L))
    torch::torch_cat(list(hidden_states, last_channel), dim = 2L)
  }
)

#' LTX-2.3 video decoder
#'
#' Latents [B, 128, F, H, W] -> pixel video [B, 3, 8F - 7, 32H, 32W].
#' Block channel lists are given encoder-side (as in the config) and
#' reversed internally; \code{upsample_type} is indexed directly.
#'
#' @param in_channels,out_channels Integers. Latent and pixel channels.
#' @param block_out_channels Integer vector (config order).
#' @param spatio_temporal_scaling Logical vector per up block.
#' @param layers_per_block Integer vector (config order; first entry is
#'   the mid block after reversal).
#' @param upsample_type Character vector per up block (not reversed).
#' @param patch_size,patch_size_t Integers.
#' @param resnet_norm_eps Numeric.
#' @param is_causal Logical. FALSE for LTX (symmetric temporal padding).
#' @param upsample_residual Logical vector per up block.
#' @param upsample_factor Integer vector per up block.
#' @param spatial_padding_mode Character.
#'
#' @export
ltx23_video_decoder3d <- torch::nn_module(
  "ltx23_video_decoder3d",
  initialize = function(
    in_channels = 128L,
    out_channels = 3L,
    block_out_channels = c(256L, 512L, 512L, 1024L),
    spatio_temporal_scaling = c(TRUE, TRUE, TRUE, TRUE),
    layers_per_block = c(4L, 6L, 4L, 2L, 2L),
    upsample_type = c("spatiotemporal", "spatiotemporal", "temporal", "spatial"),
    patch_size = 4L,
    patch_size_t = 1L,
    resnet_norm_eps = 1e-6,
    is_causal = FALSE,
    upsample_residual = c(FALSE, FALSE, FALSE, FALSE),
    upsample_factor = c(2L, 2L, 1L, 2L),
    spatial_padding_mode = "zeros"
  ) {
    self$patch_size <- as.integer(patch_size)
    self$patch_size_t <- as.integer(patch_size_t)
    self$out_channels <- out_channels * patch_size^2
    self$is_causal <- is_causal

    block_out_channels <- rev(block_out_channels)
    spatio_temporal_scaling <- rev(spatio_temporal_scaling)
    layers_per_block <- rev(layers_per_block)
    upsample_residual <- rev(upsample_residual)
    upsample_factor <- rev(upsample_factor)
    # NOTE: upsample_type is deliberately NOT reversed (reference behavior)

    output_channel <- block_out_channels[1]
    self$conv_in <- ltx23_causal_conv3d(
      in_channels, output_channel, kernel_size = 3L, stride = 1L,
      spatial_padding_mode = spatial_padding_mode
    )

    self$mid_block <- ltx23_video_mid_block3d(
      in_channels = output_channel,
      num_layers = layers_per_block[1],
      resnet_eps = resnet_norm_eps,
      spatial_padding_mode = spatial_padding_mode
    )

    up_blocks <- list()
    for (i in seq_along(block_out_channels)) {
      input_channel <- output_channel %/% upsample_factor[i]
      output_channel <- block_out_channels[i] %/% upsample_factor[i]
      up_blocks[[i]] <- ltx23_video_up_block3d(
        in_channels = input_channel,
        out_channels = output_channel,
        num_layers = layers_per_block[i + 1L],
        resnet_eps = resnet_norm_eps,
        spatio_temporal_scale = spatio_temporal_scaling[i],
        upsample_type = upsample_type[i],
        upsample_residual = upsample_residual[i],
        upscale_factor = upsample_factor[i],
        spatial_padding_mode = spatial_padding_mode
      )
    }
    self$up_blocks <- torch::nn_module_list(up_blocks)

    self$norm_out <- ltx23_per_channel_rms_norm()
    self$conv_out <- ltx23_causal_conv3d(
      output_channel, self$out_channels, kernel_size = 3L, stride = 1L,
      spatial_padding_mode = spatial_padding_mode
    )
  },
  forward = function(hidden_states, causal = NULL) {
    causal <- causal %||% self$is_causal

    hidden_states <- self$conv_in(hidden_states, causal = causal)
    hidden_states <- self$mid_block(hidden_states, causal = causal)
    for (i in seq_along(self$up_blocks)) {
      hidden_states <- self$up_blocks[[i]](hidden_states, causal = causal)
    }

    hidden_states <- self$norm_out(hidden_states)
    hidden_states <- torch::nnf_silu(hidden_states)
    hidden_states <- self$conv_out(hidden_states, causal = causal)

    # Un-patchify: channel-to-space with the LTX ordering
    p <- self$patch_size
    p_t <- self$patch_size_t
    dims <- hidden_states$shape
    hidden_states <- hidden_states$reshape(c(
      dims[1], -1L, p_t, p, p, dims[3], dims[4], dims[5]
    ))
    # [B, C, pt, ph, pw, F, H, W] -> [B, C, F, pt, H, pw, W, ph]
    hidden_states <- hidden_states$permute(c(1L, 2L, 6L, 3L, 7L, 5L, 8L, 4L))
    hidden_states$flatten(start_dim = 7L, end_dim = 8L)$
      flatten(start_dim = 5L, end_dim = 6L)$
      flatten(start_dim = 3L, end_dim = 4L)
  }
)

#' LTX-2.3 video VAE
#'
#' Encoder + decoder + per-channel latent statistics (loaded from the
#' checkpoint's \code{per_channel_statistics}). The checkpoint's
#' \code{scaling_factor} is 1.0, so latent (de)normalization is purely
#' the per-channel affine map.
#'
#' @param latent_channels Integer.
#' @param ... Passed to both encoder and decoder constructors where they
#'   accept it (see \code{\link{ltx23_video_encoder3d}} and
#'   \code{\link{ltx23_video_decoder3d}} for the full parameter set).
#'
#' @export
ltx23_video_vae <- torch::nn_module(
  "ltx23_video_vae",
  initialize = function(
    in_channels = 3L,
    out_channels = 3L,
    latent_channels = 128L,
    block_out_channels = c(256L, 512L, 1024L, 1024L),
    decoder_block_out_channels = c(256L, 512L, 512L, 1024L),
    layers_per_block = c(4L, 6L, 4L, 2L, 2L),
    decoder_layers_per_block = c(4L, 6L, 4L, 2L, 2L),
    spatio_temporal_scaling = c(TRUE, TRUE, TRUE, TRUE),
    decoder_spatio_temporal_scaling = c(TRUE, TRUE, TRUE, TRUE),
    downsample_type = c("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
    upsample_type = c("spatiotemporal", "spatiotemporal", "temporal", "spatial"),
    upsample_residual = c(FALSE, FALSE, FALSE, FALSE),
    upsample_factor = c(2L, 2L, 1L, 2L),
    patch_size = 4L,
    patch_size_t = 1L,
    resnet_norm_eps = 1e-6,
    encoder_causal = TRUE,
    decoder_causal = FALSE,
    encoder_spatial_padding_mode = "zeros",
    decoder_spatial_padding_mode = "zeros"
  ) {
    self$latent_channels <- as.integer(latent_channels)

    self$encoder <- ltx23_video_encoder3d(
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
    self$decoder <- ltx23_video_decoder3d(
      in_channels = latent_channels,
      out_channels = out_channels,
      block_out_channels = decoder_block_out_channels,
      spatio_temporal_scaling = decoder_spatio_temporal_scaling,
      layers_per_block = decoder_layers_per_block,
      upsample_type = upsample_type,
      patch_size = patch_size,
      patch_size_t = patch_size_t,
      resnet_norm_eps = resnet_norm_eps,
      is_causal = decoder_causal,
      upsample_residual = upsample_residual,
      upsample_factor = upsample_factor,
      spatial_padding_mode = decoder_spatial_padding_mode
    )

    self$latents_mean <- torch::nn_buffer(torch::torch_zeros(latent_channels))
    self$latents_std <- torch::nn_buffer(torch::torch_ones(latent_channels))
  },
  encode = function(x, causal = NULL) {
    moments <- self$encoder(x, causal = causal)
    n <- self$latent_channels
    list(
      mean = moments$narrow(2L, 1L, n),
      logvar = moments$narrow(2L, n + 1L, n)
    )
  },
  decode = function(z, causal = NULL) {
    self$decoder(z, causal = causal)
  },
  forward = function(z) {
    self$decode(z)
  }
)

#' Normalize latents with the VAE's per-channel statistics
#'
#' @param latents Tensor [B, C, F, H, W].
#' @param latents_mean,latents_std Tensors [C].
#'
#' @return Normalized latents.
#'
#' @export
ltx23_normalize_latents <- function(latents, latents_mean, latents_std) {
  mean <- latents_mean$view(c(1L, -1L, 1L, 1L, 1L))$to(device = latents$device, dtype = latents$dtype)
  std <- latents_std$view(c(1L, -1L, 1L, 1L, 1L))$to(device = latents$device, dtype = latents$dtype)
  (latents - mean) / std
}

#' Denormalize latents with the VAE's per-channel statistics
#'
#' @param latents Tensor [B, C, F, H, W].
#' @param latents_mean,latents_std Tensors [C].
#'
#' @return Denormalized latents ready for the decoder.
#'
#' @export
ltx23_denormalize_latents <- function(latents, latents_mean, latents_std) {
  mean <- latents_mean$view(c(1L, -1L, 1L, 1L, 1L))$to(device = latents$device, dtype = latents$dtype)
  std <- latents_std$view(c(1L, -1L, 1L, 1L, 1L))$to(device = latents$device, dtype = latents$dtype)
  latents * std + mean
}

#' Map an official VAE checkpoint key to the R module name
#'
#' The official checkpoint stores the encoder/decoder as flat block lists
#' (down_blocks.0-8 / up_blocks.0-8) where downsamplers/upsamplers and
#' the mid block are separate entries; diffusers (and this port) nest
#' them. Index mapping per diffusers convert_ltx2_to_diffusers.py.
#'
#' @param key Character. Checkpoint key (with or without "vae." prefix).
#'
#' @return Character. Module parameter/buffer name.
#'
#' @export
ltx23_map_vae_key <- function(key) {
  key <- sub("^vae\\.", "", key)

  key <- sub("^per_channel_statistics\\.mean-of-means$", "latents_mean", key)
  key <- sub("^per_channel_statistics\\.std-of-means$", "latents_std", key)

  down_map <- c(
    "0" = "down_blocks.0",
    "1" = "down_blocks.0.downsamplers.0",
    "2" = "down_blocks.1",
    "3" = "down_blocks.1.downsamplers.0",
    "4" = "down_blocks.2",
    "5" = "down_blocks.2.downsamplers.0",
    "6" = "down_blocks.3",
    "7" = "down_blocks.3.downsamplers.0",
    "8" = "mid_block"
  )
  up_map <- c(
    "0" = "mid_block",
    "1" = "up_blocks.0.upsamplers.0",
    "2" = "up_blocks.0",
    "3" = "up_blocks.1.upsamplers.0",
    "4" = "up_blocks.1",
    "5" = "up_blocks.2.upsamplers.0",
    "6" = "up_blocks.2",
    "7" = "up_blocks.3.upsamplers.0",
    "8" = "up_blocks.3"
  )

  m <- regmatches(key, regexec("^(encoder\\.down_blocks|decoder\\.up_blocks)\\.([0-9]+)\\.(.*)$", key))[[1]]
  if (length(m) == 4L) {
    section <- if (startsWith(m[2], "encoder")) "encoder" else "decoder"
    map <- if (section == "encoder") down_map else up_map
    idx <- m[3]
    if (is.na(map[idx])) {
      return(NA_character_)
    }
    rest <- m[4]
    # Sampler entries map to the nested module directly (their inner
    # structure is just .conv.conv / norms)
    key <- paste0(section, ".", map[[idx]], ".", rest)
  }

  key <- gsub("res_blocks", "resnets", key, fixed = TRUE)
  key
}
