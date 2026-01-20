#' Native UNet for Stable Diffusion
#'
#' Native R torch implementation of UNet2DConditionModel.
#' Replaces TorchScript for better GPU compatibility.
#'
#' @param in_channels Input channels (default 4 for latent space)
#' @param out_channels Output channels (default 4)
#' @param block_out_channels Channel multipliers per block
#' @param layers_per_block Number of ResBlocks per down/up block
#' @param cross_attention_dim Context dimension from text encoder
#' @param attention_head_dim Dimension per attention head
#'
#' @return An nn_module representing the UNet
#' @export
unet_native <- torch::nn_module(
  "UNetNative",

  initialize = function(
    in_channels = 4L,
    out_channels = 4L,
    block_out_channels = c(320L, 640L, 1280L, 1280L),
    layers_per_block = 2L,
    cross_attention_dim = 1024L,
    attention_head_dim = 64L
  ) {

    self$in_channels <- in_channels
    self$out_channels <- out_channels
    self$block_out_channels <- block_out_channels
    self$layers_per_block <- layers_per_block

    # Time embedding dimension
    time_embed_dim <- block_out_channels[1] * 4L# 320 * 4 = 1280

    # Input convolution
    self$conv_in <- torch::nn_conv2d(in_channels, block_out_channels[1], 3L, padding = 1L)

    # Time embedding MLP
    self$time_embedding_linear_1 <- torch::nn_linear(block_out_channels[1], time_embed_dim)
    self$time_embedding_linear_2 <- torch::nn_linear(time_embed_dim, time_embed_dim)

    # Down blocks
    # SD21: blocks 0-2 have attention, block 3 does not
    self$down_blocks <- torch::nn_module_list()
    output_channel <- block_out_channels[1]

    for (i in seq_along(block_out_channels)) {
      input_channel <- output_channel
      output_channel <- block_out_channels[i]
      is_final_block <- (i == length(block_out_channels))
      # Last block has no attention
      has_attention <- !is_final_block

      down_block <- self$create_down_block(
        input_channel, output_channel, time_embed_dim,
        cross_attention_dim, attention_head_dim,
        layers_per_block,
        add_downsample = !is_final_block,
        add_attention = has_attention
      )
      self$down_blocks$append(down_block)
    }

    # Mid block
    mid_channels <- block_out_channels[length(block_out_channels)]
    self$mid_block <- self$create_mid_block(
      mid_channels, time_embed_dim, cross_attention_dim, attention_head_dim
    )

    # Up blocks (reverse order)
    # SD21: first up_block (index 0) has no attention, rest do
    # Skip channels vary per resnet - we store them for forward pass
    self$up_blocks <- torch::nn_module_list()
    reversed_channels <- rev(block_out_channels)

    # Calculate skip channels for each resnet
    # The skip stack order (bottom to top):
    # down0: [in, r0, r1, ds], down1: [r0, r1, ds], down2: [r0, r1, ds], down3: [r0, r1]
    # This gives us: [320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280]
    skip_channels <- c(
      block_out_channels[1], # conv_in
      rep(block_out_channels[1], layers_per_block), # down0 resnets
      block_out_channels[1], # down0 downsample
      rep(block_out_channels[2], layers_per_block), # down1 resnets
      block_out_channels[2], # down1 downsample
      rep(block_out_channels[3], layers_per_block), # down2 resnets
      block_out_channels[3], # down2 downsample
      rep(block_out_channels[4], layers_per_block) # down3 resnets (no downsample)
    )
    self$skip_channels <- rev(skip_channels) # Reverse for popping order

    for (i in seq_along(reversed_channels)) {
      output_channel <- reversed_channels[i]
      if (i == 1) {
        prev_output <- mid_channels
      } else {
        prev_output <- reversed_channels[i - 1]
      }

      is_final_block <- (i == length(reversed_channels))
      is_first_block <- (i == 1)
      has_attention <- !is_first_block

      # Calculate per-resnet skip channels for this up_block
      num_resnets <- layers_per_block + 1L
      resnet_skip_channels <- integer(num_resnets)
      for (j in seq_len(num_resnets)) {
        skip_idx <- (i - 1) * num_resnets + j
        resnet_skip_channels[j] <- self$skip_channels[skip_idx]
      }

      up_block <- self$create_up_block_v2(
        prev_output, output_channel, resnet_skip_channels, time_embed_dim,
        cross_attention_dim, attention_head_dim,
        add_upsample = !is_final_block,
        add_attention = has_attention
      )
      self$up_blocks$append(up_block)
    }

    # Output
    self$conv_norm_out <- group_norm_32(block_out_channels[1])
    self$conv_out <- torch::nn_conv2d(block_out_channels[1], out_channels, 3L, padding = 1L)
  },

  create_down_block = function(
    in_channels,
    out_channels,
    time_embed_dim,
    cross_attention_dim,
    attention_head_dim,
    num_layers,
    add_downsample = TRUE,
    add_attention = TRUE
  ) {
    # Create block as a simple nn_module
    DownBlock <- torch::nn_module(
      "DownBlock",
      initialize = function() {
        # ResNets
        self$resnets <- torch::nn_module_list()
        for (i in seq_len(num_layers)) {
          if (i == 1) {
            in_ch <- in_channels
          } else {
            in_ch <- out_channels
          }
          self$resnets$append(UNetResBlock(in_ch, out_channels, time_embed_dim))
        }

        # Attention blocks (optional)
        self$has_attentions <- add_attention
        if (add_attention) {
          self$attentions <- torch::nn_module_list()
          n_heads <- out_channels %/% attention_head_dim
          for (i in seq_len(num_layers)) {
            self$attentions$append(
              SpatialTransformer(out_channels, n_heads, attention_head_dim,
                depth = 1L, context_dim = cross_attention_dim)
            )
          }
        }

        # Downsample
        self$has_downsamplers <- add_downsample
        if (add_downsample) {
          self$downsamplers <- torch::nn_module_list()
          self$downsamplers$append(Downsample2D(out_channels))
        }
      }
    )
    DownBlock()
  },

  create_mid_block = function(
    channels,
    time_embed_dim,
    cross_attention_dim,
    attention_head_dim
  ) {
    MidBlock <- torch::nn_module(
      "MidBlock",
      initialize = function() {
        # Two resnets with attention in between
        self$resnets <- torch::nn_module_list()
        self$resnets$append(UNetResBlock(channels, channels, time_embed_dim))
        self$resnets$append(UNetResBlock(channels, channels, time_embed_dim))

        # Attention
        n_heads <- channels %/% attention_head_dim
        self$attentions <- torch::nn_module_list()
        self$attentions$append(
          SpatialTransformer(channels, n_heads, attention_head_dim,
            depth = 1L, context_dim = cross_attention_dim)
        )
      }
    )
    MidBlock()
  },

  create_up_block_v2 = function(
    in_channels,
    out_channels,
    resnet_skip_channels,
    time_embed_dim,
    cross_attention_dim,
    attention_head_dim,
    add_upsample = TRUE,
    add_attention = TRUE
  ) {
    # resnet_skip_channels is a vector with one entry per resnet
    num_resnets <- length(resnet_skip_channels)

    UpBlock <- torch::nn_module(
      "UpBlock",
      initialize = function() {
        # ResNets (handle skip connections with variable channels)
        self$resnets <- torch::nn_module_list()
        for (i in seq_len(num_resnets)) {
          # First resnet takes prev_output, rest take out_channels
          if (i == 1) {
            in_ch <- in_channels + resnet_skip_channels[i]
          } else {
            in_ch <- out_channels + resnet_skip_channels[i]
          }
          self$resnets$append(UNetResBlock(in_ch, out_channels, time_embed_dim))
        }

        # Attention blocks (optional)
        self$has_attentions <- add_attention
        if (add_attention) {
          self$attentions <- torch::nn_module_list()
          n_heads <- out_channels %/% attention_head_dim
          for (i in seq_len(num_resnets)) {
            self$attentions$append(
              SpatialTransformer(out_channels, n_heads, attention_head_dim,
                depth = 1L, context_dim = cross_attention_dim)
            )
          }
        }

        # Upsample
        self$has_upsamplers <- add_upsample
        if (add_upsample) {
          self$upsamplers <- torch::nn_module_list()
          self$upsamplers$append(Upsample2D(out_channels))
        }
      }
    )
    UpBlock()
  },

  forward = function(
    sample,
    timestep,
    encoder_hidden_states
  ) {
    # Get model dtype from weights
    model_dtype <- self$time_embedding_linear_1$weight$dtype

    # Time embedding (computed in float32, then cast to model dtype)
    # SD21 uses flip_sin_to_cos=FALSE, downscale_freq_shift=1
    t_emb <- timestep_embedding(timestep, self$block_out_channels[1],
      flip_sin_to_cos = FALSE, downscale_freq_shift = 1L)
    t_emb <- t_emb$to(dtype = model_dtype)
    t_emb <- self$time_embedding_linear_1(t_emb)
    t_emb <- torch::nnf_silu(t_emb)
    t_emb <- self$time_embedding_linear_2(t_emb)

    # Input conv
    sample <- self$conv_in(sample)

    # Store skip connections
    down_block_res_samples <- list(sample)

    # Down blocks
    for (i in seq_along(self$down_blocks)) {
      block <- self$down_blocks[[i]]

      for (j in seq_along(block$resnets)) {
        sample <- block$resnets[[j]](sample, t_emb)
        if (block$has_attentions) {
          sample <- block$attentions[[j]](sample, encoder_hidden_states)
        }
        down_block_res_samples <- c(down_block_res_samples, list(sample))
      }

      if (block$has_downsamplers) {
        sample <- block$downsamplers[[1]](sample)
        down_block_res_samples <- c(down_block_res_samples, list(sample))
      }
    }

    # Mid block
    sample <- self$mid_block$resnets[[1]](sample, t_emb)
    sample <- self$mid_block$attentions[[1]](sample, encoder_hidden_states)
    sample <- self$mid_block$resnets[[2]](sample, t_emb)

    # Up blocks
    for (i in seq_along(self$up_blocks)) {
      block <- self$up_blocks[[i]]

      for (j in seq_along(block$resnets)) {
        # Pop skip connection
        res_sample <- down_block_res_samples[[length(down_block_res_samples)]]
        down_block_res_samples <- down_block_res_samples[- length(down_block_res_samples)]

        # Concatenate
        sample <- torch::torch_cat(list(sample, res_sample), dim = 2L)

        sample <- block$resnets[[j]](sample, t_emb)
        if (block$has_attentions) {
          sample <- block$attentions[[j]](sample, encoder_hidden_states)
        }
      }

      if (block$has_upsamplers) {
        sample <- block$upsamplers[[1]](sample)
      }
    }

    # Output
    sample <- self$conv_norm_out(sample)
    sample <- torch::nnf_silu(sample)
    sample <- self$conv_out(sample)

    sample
  }
)

#' Detect UNet architecture from TorchScript file
#'
#' @param torchscript_path Path to TorchScript UNet .pt file
#' @return List with architecture parameters
#' @keywords internal
detect_unet_architecture <- function(torchscript_path) {
  ts_unet <- torch::jit_load(torchscript_path)
  ts_params <- ts_unet$parameters
  param_names <- names(ts_params)

  # Get block_out_channels from down_blocks resnets
  # Use norm2 (not norm1) to get output channels
  block_channels <- integer(0)
  for (i in 0:3) {
    key <- sprintf("unet.down_blocks.%d.resnets.0.norm2.weight", i)
    if (key %in% param_names) {
      block_channels <- c(block_channels, as.integer(ts_params[[key]]$shape[1]))
    }
  }

  # Get cross_attention_dim from attn2.to_k
  attn_key <- "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight"
  cross_attention_dim <- as.integer(ts_params[[attn_key]]$shape[2])

  # attention_head_dim is a fixed architectural choice for SD models
  # SD21 and SDXL use 64
  attention_head_dim <- 64L

  list(
    in_channels = 4L,
    out_channels = 4L,
    block_out_channels = block_channels,
    layers_per_block = 2L,
    cross_attention_dim = cross_attention_dim,
    attention_head_dim = attention_head_dim
  )
}

#' Create native UNet from TorchScript
#'
#' Detects architecture and loads weights from a TorchScript UNet file.
#'
#' @param torchscript_path Path to TorchScript UNet .pt file
#' @param verbose Print loading progress
#'
#' @return A native UNet module with loaded weights
#' @export
unet_native_from_torchscript <- function(
  torchscript_path,
  verbose = TRUE
) {
  # Detect architecture
  arch <- detect_unet_architecture(torchscript_path)

  if (verbose) {
    message("Detected UNet architecture:")
    message("  block_out_channels: ", paste(arch$block_out_channels, collapse = ", "))
    message("  cross_attention_dim: ", arch$cross_attention_dim)
    message("  attention_head_dim: ", arch$attention_head_dim)
  }

  # Create native UNet
  unet <- unet_native(
    in_channels = arch$in_channels,
    out_channels = arch$out_channels,
    block_out_channels = arch$block_out_channels,
    layers_per_block = arch$layers_per_block,
    cross_attention_dim = arch$cross_attention_dim,
    attention_head_dim = arch$attention_head_dim
  )

  # Load weights
  load_unet_weights(unet, torchscript_path, verbose = verbose)

  unet
}

#' Load weights from TorchScript UNet into native UNet
#'
#' @param native_unet Native UNet module
#' @param torchscript_path Path to TorchScript UNet .pt file
#' @param verbose Print loading progress
#'
#' @return The native UNet with loaded weights (invisibly)
#' @export
load_unet_weights <- function(
  native_unet,
  torchscript_path,
  verbose = TRUE
) {
  ts_unet <- torch::jit_load(torchscript_path)
  ts_params <- ts_unet$parameters

  # Build weight mapping
  remap_key <- function(key) {
    # Strip unet. prefix
    key <- sub("^unet\\.", "", key)

    # Time embedding: time_embedding.linear_1 -> time_embedding_linear_1
    key <- sub("^time_embedding\\.linear_1", "time_embedding_linear_1", key)
    key <- sub("^time_embedding\\.linear_2", "time_embedding_linear_2", key)

    # Note: nn_sequential children use 0-indexing (matching Python/TorchScript)
    # so no conversion needed for to_out.0, net.0, net.2

    key
  }

  loaded <- 0
  skipped <- 0
  unmapped <- character(0)

  torch::with_no_grad({
      for (ts_name in names(ts_params)) {
        native_name <- remap_key(ts_name)

        if (native_name %in% names(native_unet$parameters)) {
          ts_tensor <- ts_params[[ts_name]]
          native_tensor <- native_unet$parameters[[native_name]]

          if (all(ts_tensor$shape == native_tensor$shape)) {
            native_tensor$copy_(ts_tensor)
            loaded <- loaded + 1
          } else if (verbose) {
            message("Shape mismatch: ", native_name,
              " (", paste(as.integer(ts_tensor$shape), collapse = "x"), " vs ",
              paste(as.integer(native_tensor$shape), collapse = "x"), ")")
            skipped <- skipped + 1
          }
        } else {
          skipped <- skipped + 1
          unmapped <- c(unmapped, paste0(ts_name, " -> ", native_name))
        }
      }
    })

  if (verbose) {
    if (length(unmapped) > 0 && length(unmapped) <= 10) {
      message("Unmapped parameters:")
      for (u in unmapped) message("  ", u)
    } else if (length(unmapped) > 10) {
      message("Unmapped: ", length(unmapped), " parameters (showing first 10)")
      for (u in head(unmapped, 10)) message("  ", u)
    }
    message("Loaded ", loaded, "/", loaded + skipped, " parameters")
  }

  invisible(native_unet)
}

#' Native SDXL UNet
#'
#' Native R torch implementation of SDXL UNet2DConditionModel.
#' SDXL has a different architecture from SD21:
#' - 3 down/up blocks (not 4)
#' - Variable transformer depth per block
#' - Additional conditioning via add_embedding
#'
#' @param in_channels Input channels (default 4 for latent space)
#' @param out_channels Output channels (default 4)
#' @param block_out_channels Channel multipliers per block
#' @param layers_per_block Number of ResBlocks per down/up block
#' @param transformer_layers_per_block Transformer depth per block
#' @param cross_attention_dim Context dimension from text encoder
#' @param attention_head_dim Dimension per attention head
#' @param addition_embed_dim Dimension for additional embeddings
#' @param addition_time_embed_dim Dimension for time embedding projection
#'
#' @return An nn_module representing the SDXL UNet
#' @export
unet_sdxl_native <- torch::nn_module(
  "UNetSDXLNative",

  initialize = function(
    in_channels = 4L,
    out_channels = 4L,
    block_out_channels = c(320L, 640L, 1280L),
    layers_per_block = 2L,
    transformer_layers_per_block = c(0L, 2L, 10L),
    cross_attention_dim = 2048L,
    attention_head_dim = 64L,
    addition_embed_dim = 1280L,
    addition_time_embed_dim = 256L
  ) {

    self$in_channels <- in_channels
    self$out_channels <- out_channels
    self$block_out_channels <- block_out_channels
    self$layers_per_block <- layers_per_block
    self$transformer_layers_per_block <- transformer_layers_per_block

    # Time embedding dimension (same as SD21)
    time_embed_dim <- block_out_channels[1] * 4L# 320 * 4 = 1280

    # Input convolution
    self$conv_in <- torch::nn_conv2d(in_channels, block_out_channels[1], 3L, padding = 1L)

    # Time embedding MLP
    self$time_embedding_linear_1 <- torch::nn_linear(block_out_channels[1], time_embed_dim)
    self$time_embedding_linear_2 <- torch::nn_linear(time_embed_dim, time_embed_dim)

    # Additional embedding (SDXL-specific): projects text_embeds + time_ids
    # time_ids has 6 values, projected through 256-dim Fourier features = 1536
    # text_embeds is 1280-dim, so total input is 2816
    add_embed_input_dim <- addition_embed_dim + 6L * addition_time_embed_dim
    self$add_embedding_linear_1 <- torch::nn_linear(add_embed_input_dim, time_embed_dim)
    self$add_embedding_linear_2 <- torch::nn_linear(time_embed_dim, time_embed_dim)

    # Time projection for time_ids (SDXL uses Fourier features)
    self$add_time_proj_dim <- addition_time_embed_dim

    # Down blocks (SDXL: 3 blocks with variable attention)
    self$down_blocks <- torch::nn_module_list()
    output_channel <- block_out_channels[1]

    for (i in seq_along(block_out_channels)) {
      input_channel <- output_channel
      output_channel <- block_out_channels[i]
      is_final_block <- (i == length(block_out_channels))
      transformer_depth <- transformer_layers_per_block[i]

      down_block <- self$create_down_block_sdxl(
        input_channel, output_channel, time_embed_dim,
        cross_attention_dim, attention_head_dim,
        layers_per_block, transformer_depth,
        add_downsample = !is_final_block
      )
      self$down_blocks$append(down_block)
    }

    # Mid block (SDXL: 10 transformer blocks)
    mid_channels <- block_out_channels[length(block_out_channels)]
    mid_transformer_depth <- transformer_layers_per_block[length(transformer_layers_per_block)]
    self$mid_block <- self$create_mid_block_sdxl(
      mid_channels, time_embed_dim, cross_attention_dim, attention_head_dim,
      mid_transformer_depth
    )

    # Up blocks (SDXL: reverse order, 3 blocks)
    self$up_blocks <- torch::nn_module_list()
    reversed_channels <- rev(block_out_channels)
    reversed_transformer_depths <- rev(transformer_layers_per_block)

    # Calculate skip channels for SDXL (3 blocks)
    # down0: [in, r0, r1, ds], down1: [r0, r1, ds], down2: [r0, r1]
    skip_channels <- c(
      block_out_channels[1], # conv_in
      rep(block_out_channels[1], layers_per_block), # down0 resnets
      block_out_channels[1], # down0 downsample
      rep(block_out_channels[2], layers_per_block), # down1 resnets
      block_out_channels[2], # down1 downsample
      rep(block_out_channels[3], layers_per_block) # down2 resnets (no downsample)
    )
    self$skip_channels <- rev(skip_channels)

    for (i in seq_along(reversed_channels)) {
      output_channel <- reversed_channels[i]
      if (i == 1) {
        prev_output <- mid_channels
      } else {
        prev_output <- reversed_channels[i - 1]
      }

      is_final_block <- (i == length(reversed_channels))
      transformer_depth <- reversed_transformer_depths[i]

      # Calculate per-resnet skip channels
      num_resnets <- layers_per_block + 1L
      resnet_skip_channels <- integer(num_resnets)
      for (j in seq_len(num_resnets)) {
        skip_idx <- (i - 1) * num_resnets + j
        resnet_skip_channels[j] <- self$skip_channels[skip_idx]
      }

      up_block <- self$create_up_block_sdxl(
        prev_output, output_channel, resnet_skip_channels, time_embed_dim,
        cross_attention_dim, attention_head_dim, transformer_depth,
        add_upsample = !is_final_block
      )
      self$up_blocks$append(up_block)
    }

    # Output
    self$conv_norm_out <- group_norm_32(block_out_channels[1])
    self$conv_out <- torch::nn_conv2d(block_out_channels[1], out_channels, 3L, padding = 1L)
  },

  create_down_block_sdxl = function(
    in_channels,
    out_channels,
    time_embed_dim,
    cross_attention_dim,
    attention_head_dim,
    num_layers,
    transformer_depth,
    add_downsample = TRUE
  ) {
    DownBlock <- torch::nn_module(
      "DownBlockSDXL",
      initialize = function() {
        # ResNets
        self$resnets <- torch::nn_module_list()
        for (i in seq_len(num_layers)) {
          if (i == 1) {
            in_ch <- in_channels
          } else {
            in_ch <- out_channels
          }
          self$resnets$append(UNetResBlock(in_ch, out_channels, time_embed_dim))
        }

        # Attention blocks (only if transformer_depth > 0)
        self$has_attentions <- transformer_depth > 0
        if (self$has_attentions) {
          self$attentions <- torch::nn_module_list()
          n_heads <- out_channels %/% attention_head_dim
          for (i in seq_len(num_layers)) {
            self$attentions$append(
              SpatialTransformer(out_channels, n_heads, attention_head_dim,
                depth = transformer_depth, context_dim = cross_attention_dim)
            )
          }
        }

        # Downsample
        self$has_downsamplers <- add_downsample
        if (add_downsample) {
          self$downsamplers <- torch::nn_module_list()
          self$downsamplers$append(Downsample2D(out_channels))
        }
      }
    )
    DownBlock()
  },

  create_mid_block_sdxl = function(
    channels,
    time_embed_dim,
    cross_attention_dim,
    attention_head_dim,
    transformer_depth
  ) {
    MidBlock <- torch::nn_module(
      "MidBlockSDXL",
      initialize = function() {
        self$resnets <- torch::nn_module_list()
        self$resnets$append(UNetResBlock(channels, channels, time_embed_dim))
        self$resnets$append(UNetResBlock(channels, channels, time_embed_dim))

        n_heads <- channels %/% attention_head_dim
        self$attentions <- torch::nn_module_list()
        self$attentions$append(
          SpatialTransformer(channels, n_heads, attention_head_dim,
            depth = transformer_depth, context_dim = cross_attention_dim)
        )
      }
    )
    MidBlock()
  },

  create_up_block_sdxl = function(
    in_channels,
    out_channels,
    resnet_skip_channels,
    time_embed_dim,
    cross_attention_dim,
    attention_head_dim,
    transformer_depth,
    add_upsample = TRUE
  ) {
    num_resnets <- length(resnet_skip_channels)

    UpBlock <- torch::nn_module(
      "UpBlockSDXL",
      initialize = function() {
        self$resnets <- torch::nn_module_list()
        for (i in seq_len(num_resnets)) {
          if (i == 1) {
            in_ch <- in_channels + resnet_skip_channels[i]
          } else {
            in_ch <- out_channels + resnet_skip_channels[i]
          }
          self$resnets$append(UNetResBlock(in_ch, out_channels, time_embed_dim))
        }

        # Attention blocks (only if transformer_depth > 0)
        self$has_attentions <- transformer_depth > 0
        if (self$has_attentions) {
          self$attentions <- torch::nn_module_list()
          n_heads <- out_channels %/% attention_head_dim
          for (i in seq_len(num_resnets)) {
            self$attentions$append(
              SpatialTransformer(out_channels, n_heads, attention_head_dim,
                depth = transformer_depth, context_dim = cross_attention_dim)
            )
          }
        }

        # Upsample
        self$has_upsamplers <- add_upsample
        if (add_upsample) {
          self$upsamplers <- torch::nn_module_list()
          self$upsamplers$append(Upsample2D(out_channels))
        }
      }
    )
    UpBlock()
  },

  forward = function(
    sample,
    timestep,
    encoder_hidden_states,
    text_embeds,
    time_ids
  ) {
    # Accepts same signature as TorchScript:
    # unet(sample, timestep, encoder_hidden_states, text_embeds, time_ids)

    # Get model dtype from weights
    model_dtype <- self$time_embedding_linear_1$weight$dtype

    # Time embedding (computed in float32, then cast to model dtype)
    # SDXL uses flip_sin_to_cos=TRUE (default), downscale_freq_shift=0 (default)
    t_emb <- timestep_embedding(timestep, self$block_out_channels[1])
    t_emb <- t_emb$to(dtype = model_dtype)
    t_emb <- self$time_embedding_linear_1(t_emb)
    t_emb <- torch::nnf_silu(t_emb)
    t_emb <- self$time_embedding_linear_2(t_emb)

    # Additional embedding (Fourier projection of time_ids + text_embeds)
    # Uses same defaults as main time embedding
    time_ids_emb <- timestep_embedding(time_ids$flatten(), self$add_time_proj_dim)
    time_ids_emb <- time_ids_emb$to(dtype = model_dtype)
    time_ids_emb <- time_ids_emb$reshape(c(text_embeds$shape[1], - 1L))
    add_embeds <- torch::torch_cat(list(text_embeds, time_ids_emb), dim = 2L)
    add_emb <- self$add_embedding_linear_1(add_embeds)
    add_emb <- torch::nnf_silu(add_emb)
    add_emb <- self$add_embedding_linear_2(add_emb)

    # Combine time and additional embeddings
    emb <- t_emb + add_emb

    # Input conv
    sample <- self$conv_in(sample)

    # Store skip connections
    down_block_res_samples <- list(sample)

    # Down blocks
    for (i in seq_along(self$down_blocks)) {
      block <- self$down_blocks[[i]]

      for (j in seq_along(block$resnets)) {
        sample <- block$resnets[[j]](sample, emb)
        if (block$has_attentions) {
          sample <- block$attentions[[j]](sample, encoder_hidden_states)
        }
        down_block_res_samples <- c(down_block_res_samples, list(sample))
      }

      if (block$has_downsamplers) {
        sample <- block$downsamplers[[1]](sample)
        down_block_res_samples <- c(down_block_res_samples, list(sample))
      }
    }

    # Mid block
    sample <- self$mid_block$resnets[[1]](sample, emb)
    sample <- self$mid_block$attentions[[1]](sample, encoder_hidden_states)
    sample <- self$mid_block$resnets[[2]](sample, emb)

    # Up blocks
    for (i in seq_along(self$up_blocks)) {
      block <- self$up_blocks[[i]]

      for (j in seq_along(block$resnets)) {
        # Pop skip connection
        res_sample <- down_block_res_samples[[length(down_block_res_samples)]]
        down_block_res_samples <- down_block_res_samples[- length(down_block_res_samples)]

        # Concatenate
        sample <- torch::torch_cat(list(sample, res_sample), dim = 2L)

        sample <- block$resnets[[j]](sample, emb)
        if (block$has_attentions) {
          sample <- block$attentions[[j]](sample, encoder_hidden_states)
        }
      }

      if (block$has_upsamplers) {
        sample <- block$upsamplers[[1]](sample)
      }
    }

    # Output
    sample <- self$conv_norm_out(sample)
    sample <- torch::nnf_silu(sample)
    sample <- self$conv_out(sample)

    sample
  }
)

#' Detect SDXL UNet architecture from TorchScript file
#'
#' @param torchscript_path Path to TorchScript SDXL UNet .pt file
#' @return List with architecture parameters
#' @keywords internal
detect_unet_sdxl_architecture <- function(torchscript_path) {
  ts_unet <- torch::jit_load(torchscript_path, device = "cpu")
  ts_params <- ts_unet$parameters
  param_names <- names(ts_params)

  # Get block_out_channels from down_blocks resnets
  block_channels <- integer(0)
  for (i in 0:2) { # SDXL has 3 blocks
    key <- sprintf("unet.down_blocks.%d.resnets.0.norm2.weight", i)
    if (key %in% param_names) {
      block_channels <- c(block_channels, as.integer(ts_params[[key]]$shape[1]))
    }
  }

  # Get transformer depths per block
  transformer_depths <- integer(length(block_channels))
  for (i in seq_along(block_channels)) {
    depth <- 0L
    for (j in 0:15) {
      key <- sprintf("unet.down_blocks.%d.attentions.0.transformer_blocks.%d.attn1.to_q.weight", i - 1, j)
      if (key %in% param_names) depth <- depth + 1L
    }
    transformer_depths[i] <- depth
  }

  # Get cross_attention_dim from attn2.to_k (use block with attention)
  cross_attention_dim <- NULL
  for (i in 0:2) {
    attn_key <- sprintf("unet.down_blocks.%d.attentions.0.transformer_blocks.0.attn2.to_k.weight", i)
    if (attn_key %in% param_names) {
      cross_attention_dim <- as.integer(ts_params[[attn_key]]$shape[2])
      break
    }
  }

  list(
    in_channels = 4L,
    out_channels = 4L,
    block_out_channels = block_channels,
    layers_per_block = 2L,
    transformer_layers_per_block = transformer_depths,
    cross_attention_dim = cross_attention_dim,
    attention_head_dim = 64L,
    addition_embed_dim = 1280L,
    addition_time_embed_dim = 256L
  )
}

#' Create native SDXL UNet from TorchScript
#'
#' @param torchscript_path Path to TorchScript SDXL UNet .pt file
#' @param verbose Print loading progress
#'
#' @return A native SDXL UNet module with loaded weights
#' @export
unet_sdxl_native_from_torchscript <- function(
  torchscript_path,
  verbose = TRUE
) {
  arch <- detect_unet_sdxl_architecture(torchscript_path)

  if (verbose) {
    message("Detected SDXL UNet architecture:")
    message("  block_out_channels: ", paste(arch$block_out_channels, collapse = ", "))
    message("  transformer_layers_per_block: ", paste(arch$transformer_layers_per_block, collapse = ", "))
    message("  cross_attention_dim: ", arch$cross_attention_dim)
  }

  unet <- unet_sdxl_native(
    in_channels = arch$in_channels,
    out_channels = arch$out_channels,
    block_out_channels = arch$block_out_channels,
    layers_per_block = arch$layers_per_block,
    transformer_layers_per_block = arch$transformer_layers_per_block,
    cross_attention_dim = arch$cross_attention_dim,
    attention_head_dim = arch$attention_head_dim,
    addition_embed_dim = arch$addition_embed_dim,
    addition_time_embed_dim = arch$addition_time_embed_dim
  )

  load_unet_sdxl_weights(unet, torchscript_path, verbose = verbose)

  unet
}

#' Load weights from TorchScript SDXL UNet into native SDXL UNet
#'
#' @param native_unet Native SDXL UNet module
#' @param torchscript_path Path to TorchScript SDXL UNet .pt file
#' @param verbose Print loading progress
#'
#' @return The native UNet with loaded weights (invisibly)
#' @export
load_unet_sdxl_weights <- function(
  native_unet,
  torchscript_path,
  verbose = TRUE
) {
  ts_unet <- torch::jit_load(torchscript_path, device = "cpu")
  ts_params <- ts_unet$parameters

  remap_key <- function(key) {
    # Strip unet. prefix
    key <- sub("^unet\\.", "", key)

    # Time embedding: time_embedding.linear_1 -> time_embedding_linear_1
    key <- sub("^time_embedding\\.linear_1", "time_embedding_linear_1", key)
    key <- sub("^time_embedding\\.linear_2", "time_embedding_linear_2", key)

    # Add embedding: add_embedding.linear_1 -> add_embedding_linear_1
    key <- sub("^add_embedding\\.linear_1", "add_embedding_linear_1", key)
    key <- sub("^add_embedding\\.linear_2", "add_embedding_linear_2", key)

    key
  }

  loaded <- 0
  skipped <- 0
  unmapped <- character(0)

  torch::with_no_grad({
      for (ts_name in names(ts_params)) {
        native_name <- remap_key(ts_name)

        if (native_name %in% names(native_unet$parameters)) {
          ts_tensor <- ts_params[[ts_name]]
          native_tensor <- native_unet$parameters[[native_name]]

          if (all(ts_tensor$shape == native_tensor$shape)) {
            native_tensor$copy_(ts_tensor)
            loaded <- loaded + 1
          } else if (verbose) {
            message("Shape mismatch: ", native_name,
              " (", paste(as.integer(ts_tensor$shape), collapse = "x"), " vs ",
              paste(as.integer(native_tensor$shape), collapse = "x"), ")")
            skipped <- skipped + 1
          }
        } else {
          skipped <- skipped + 1
          unmapped <- c(unmapped, paste0(ts_name, " -> ", native_name))
        }
      }
    })

  if (verbose) {
    if (length(unmapped) > 0 && length(unmapped) <= 10) {
      message("Unmapped parameters:")
      for (u in unmapped) message("  ", u)
    } else if (length(unmapped) > 10) {
      message("Unmapped: ", length(unmapped), " parameters (showing first 10)")
      for (u in head(unmapped, 10)) message("  ", u)
    }
    message("Loaded ", loaded, "/", loaded + skipped, " parameters")
  }

  invisible(native_unet)
}

