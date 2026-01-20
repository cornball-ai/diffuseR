# LTX2 Text Encoder and Connectors
#
# This file implements:
# 1. LTX2TextConnectors - Transforms text embeddings for video/audio streams
# 2. Text encoder wrapper - Supports pre-computed embeddings and API-based encoding

# -----------------------------------------------------------------------------
# 1D RoPE for text connectors
# -----------------------------------------------------------------------------

#' 1D Rotary Position Embeddings for LTX2 Text Connectors
#' @keywords internal
ltx2_rotary_pos_embed_1d <- torch::nn_module(
  "LTX2RotaryPosEmbed1d",
  initialize = function(
    dim,
    base_seq_len = 4096L,
    theta = 10000.0,
    double_precision = TRUE,
    rope_type = "interleaved",
    num_attention_heads = 32L
  ) {
    self$dim <- dim
    self$base_seq_len <- base_seq_len
    self$theta <- theta
    self$double_precision <- double_precision
    self$rope_type <- rope_type
    self$num_attention_heads <- num_attention_heads
  },

  forward = function(
    batch_size,
    seq_len,
    device
  ) {
    # 1. Get 1D position ids as fractions of base_seq_len
    grid_1d <- torch::torch_arange(start = 0, end = seq_len - 1L,
      dtype = torch::torch_float32(), device = device)
    grid_1d <- grid_1d / self$base_seq_len
    grid <- grid_1d$unsqueeze(1L) $`repeat`(c(batch_size, 1L)) # [batch_size, seq_len]

    # 2. Calculate 1D RoPE frequencies
    num_rope_elems <- 2L# 1D * 2 (for cos, sin)
    if (self$double_precision) {
      freqs_dtype <- torch::torch_float64()
    } else {
      freqs_dtype <- torch::torch_float32()
    }

    pow_indices <- torch::torch_pow(
      self$theta,
      torch::torch_linspace(start = 0.0, end = 1.0, steps = self$dim %/% num_rope_elems,
        dtype = freqs_dtype, device = device)
    )
    freqs <- (pow_indices * pi / 2.0) $to(dtype = torch::torch_float32())

    # 3. Outer product: [batch_size, seq_len] x [dim/2] -> [batch_size, seq_len, dim/2]
    freqs_outer <- torch::torch_einsum("bs,d->bsd", list(grid, freqs))

    # 4. Compute cos and sin
    cos_freqs <- torch::torch_cos(freqs_outer)
    sin_freqs <- torch::torch_sin(freqs_outer)

    # 5. Interleave or split based on rope_type
    if (self$rope_type == "interleaved") {
      # Repeat each element: [B, S, D/2] -> [B, S, D]
      cos_freqs <- cos_freqs$unsqueeze(- 1L) $`repeat`(c(1L, 1L, 1L, 2L)) $flatten(start_dim = 3L)
      sin_freqs <- sin_freqs$unsqueeze(- 1L) $`repeat`(c(1L, 1L, 1L, 2L)) $flatten(start_dim = 3L)
    } else {
      # Concatenate: [B, S, D/2] -> [B, S, D]
      cos_freqs <- torch::torch_cat(list(cos_freqs, cos_freqs), dim = - 1L)
      sin_freqs <- torch::torch_cat(list(sin_freqs, sin_freqs), dim = - 1L)
    }

    list(cos_freqs, sin_freqs)
  }
)

# -----------------------------------------------------------------------------
# 1D Transformer Block for Connectors
# -----------------------------------------------------------------------------

#' 1D Transformer Block for LTX2 Text Connectors
#' @keywords internal
ltx2_transformer_block_1d <- torch::nn_module(
  "LTX2TransformerBlock1d",
  initialize = function(
    dim,
    num_attention_heads,
    attention_head_dim,
    activation_fn = "gelu-approximate",
    eps = 1e-6,
    rope_type = "interleaved"
  ) {
    self$norm1 <- rms_norm(dim, eps = eps)
    self$attn1 <- ltx2_attention(
      query_dim = dim,
      heads = num_attention_heads,
      kv_heads = num_attention_heads,
      dim_head = attention_head_dim,
      rope_type = rope_type
    )

    self$norm2 <- rms_norm(dim, eps = eps)
    self$ff <- feed_forward(dim, mult = 4L, activation_fn = activation_fn)
  },

  forward = function(
    hidden_states,
    attention_mask = NULL,
    rotary_emb = NULL
  ) {
    norm_hidden_states <- self$norm1(hidden_states)
    attn_hidden_states <- self$attn1(norm_hidden_states,
      attention_mask = attention_mask,
      query_rotary_emb = rotary_emb)
    hidden_states <- hidden_states + attn_hidden_states

    norm_hidden_states <- self$norm2(hidden_states)
    ff_hidden_states <- self$ff(norm_hidden_states)
    hidden_states <- hidden_states + ff_hidden_states

    hidden_states
  }
)

# -----------------------------------------------------------------------------
# Connector Transformer 1D
# -----------------------------------------------------------------------------

#' 1D Connector Transformer for LTX2
#' @keywords internal
ltx2_connector_transformer_1d <- torch::nn_module(
  "LTX2ConnectorTransformer1d",
  initialize = function(
    num_attention_heads = 30L,
    attention_head_dim = 128L,
    num_layers = 2L,
    num_learnable_registers = 128L,
    rope_base_seq_len = 4096L,
    rope_theta = 10000.0,
    rope_double_precision = TRUE,
    eps = 1e-6,
    causal_temporal_positioning = FALSE,
    rope_type = "interleaved"
  ) {
    self$num_attention_heads <- num_attention_heads
    self$inner_dim <- num_attention_heads * attention_head_dim
    self$causal_temporal_positioning <- causal_temporal_positioning
    self$num_learnable_registers <- num_learnable_registers

    # Learnable registers (replaces padding tokens)
    if (!is.null(num_learnable_registers) && num_learnable_registers > 0L) {
      init_registers <- torch::torch_rand(c(num_learnable_registers, self$inner_dim)) * 2.0 - 1.0
      self$learnable_registers <- torch::nn_parameter(init_registers)
    } else {
      self$learnable_registers <- NULL
    }

    # 1D RoPE
    self$rope <- ltx2_rotary_pos_embed_1d(
      dim = self$inner_dim,
      base_seq_len = rope_base_seq_len,
      theta = rope_theta,
      double_precision = rope_double_precision,
      rope_type = rope_type,
      num_attention_heads = num_attention_heads
    )

    # Transformer blocks
    self$transformer_blocks <- torch::nn_module_list(lapply(seq_len(num_layers), function(i) {
          ltx2_transformer_block_1d(
            dim = self$inner_dim,
            num_attention_heads = num_attention_heads,
            attention_head_dim = attention_head_dim,
            rope_type = rope_type
          )
        }))

    self$norm_out <- rms_norm(self$inner_dim, eps = eps)
  },

  forward = function(
    hidden_states,
    attention_mask = NULL,
    attn_mask_binarize_threshold = - 9000.0
  ) {
    batch_size <- hidden_states$shape[1]
    sequence_length <- hidden_states$shape[2]

    # 1. Replace padding with learned registers, if using
    if (!is.null(self$learnable_registers)) {
      if (sequence_length %% self$num_learnable_registers != 0L) {
        stop(sprintf("Sequence length %d must be divisible by num_learnable_registers %d",
            sequence_length, self$num_learnable_registers))
      }

      num_register_repeats <- sequence_length %/% self$num_learnable_registers
      registers <- self$learnable_registers$`repeat`(c(num_register_repeats, 1L)) # [seq_len, inner_dim]

      # Binarize attention mask
      binary_attn_mask <- (attention_mask >= attn_mask_binarize_threshold) $to(dtype = torch::torch_int32())
      if (binary_attn_mask$ndim == 4L) {
        binary_attn_mask <- binary_attn_mask$squeeze(2L) $squeeze(2L) # [B, 1, 1, L] -> [B, L]
      }

      # Extract non-padded tokens and re-pad with registers
      padded_list <- list()
      valid_seq_lens <- numeric(batch_size)

      for (i in seq_len(batch_size)) {
        mask_i <- binary_attn_mask[i,]$to(dtype = torch::torch_bool())
        hs_i <- hidden_states[i, mask_i,]
        valid_len <- as.integer(hs_i$shape[1])
        valid_seq_lens[i] <- valid_len
        pad_len <- sequence_length - valid_len

        if (pad_len > 0L) {
          # Pad with zeros on the right
          hs_i <- torch::nnf_pad(hs_i, c(0L, 0L, 0L, pad_len))
        }
        padded_list[[i]] <- hs_i$unsqueeze(1L)
      }

      padded_hidden_states <- torch::torch_cat(padded_list, dim = 1L) # [B, L, D]

      # Flip mask along sequence dimension and blend with registers
      # In R torch, flip requires a vector for dims
      flipped_mask <- torch::torch_flip(binary_attn_mask, c(2L)) $unsqueeze(- 1L) $to(dtype = hidden_states$dtype) # [B, L, 1]
      # Expand registers to batch dimension for broadcasting
      registers_expanded <- registers$unsqueeze(1L) # [L, D] -> [1, L, D] - broadcasts to [B, L, D]
      hidden_states <- flipped_mask * padded_hidden_states + (1 - flipped_mask) * registers_expanded

      # Zero out attention mask when using registers
      attention_mask <- torch::torch_zeros_like(attention_mask)
    }

    # 2. Calculate 1D RoPE
    rotary_emb <- self$rope(batch_size, sequence_length, device = hidden_states$device)

    # 3. Run transformer blocks
    for (i in seq_along(self$transformer_blocks)) {
      block <- self$transformer_blocks[[i]]
      hidden_states <- block(hidden_states, attention_mask = attention_mask, rotary_emb = rotary_emb)
    }

    hidden_states <- self$norm_out(hidden_states)

    list(hidden_states, attention_mask)
  }
)

# -----------------------------------------------------------------------------
# Full Text Connectors
# -----------------------------------------------------------------------------

#' LTX2 Text Connectors
#'
#' Transforms packed text encoder hidden states for video and audio streams.
#'
#' @param caption_channels Integer. Dimension of caption embeddings (default 3840).
#' @param text_proj_in_factor Integer. Factor for input projection (default 1).
#' @param video_connector_num_attention_heads Integer. Number of attention heads for video connector.
#' @param video_connector_attention_head_dim Integer. Attention head dimension for video.
#' @param video_connector_num_layers Integer. Number of transformer layers for video.
#' @param video_connector_num_learnable_registers Integer. Number of learnable registers for video.
#' @param audio_connector_num_attention_heads Integer. Number of attention heads for audio connector.
#' @param audio_connector_attention_head_dim Integer. Attention head dimension for audio.
#' @param audio_connector_num_layers Integer. Number of transformer layers for audio.
#' @param audio_connector_num_learnable_registers Integer. Number of learnable registers for audio.
#' @param connector_rope_base_seq_len Integer. Base sequence length for RoPE.
#' @param rope_theta Numeric. RoPE theta parameter.
#' @param rope_double_precision Logical. Use double precision for RoPE.
#' @param causal_temporal_positioning Logical. Use causal temporal positioning.
#' @param rope_type Character. RoPE type ("interleaved" or "split").
#'
#' @return nn_module for text connectors.
#' @export
ltx2_text_connectors <- torch::nn_module(
  "LTX2TextConnectors",
  initialize = function(
    caption_channels = 3840L,
    text_proj_in_factor = 49L,
    video_connector_num_attention_heads = 30L,
    video_connector_attention_head_dim = 128L,
    video_connector_num_layers = 2L,
    video_connector_num_learnable_registers = NULL,
    audio_connector_num_attention_heads = 30L,
    audio_connector_attention_head_dim = 128L,
    audio_connector_num_layers = 2L,
    audio_connector_num_learnable_registers = NULL,
    connector_rope_base_seq_len = 4096L,
    rope_theta = 10000.0,
    rope_double_precision = TRUE,
    causal_temporal_positioning = FALSE,
    rope_type = "split"
  ) {

    self$caption_channels <- caption_channels

    # Input projection (projects packed embeddings to caption_channels)
    self$text_proj_in <- torch::nn_linear(
      in_features = caption_channels * text_proj_in_factor,
      out_features = caption_channels,
      bias = FALSE
    )

    # Video connector
    self$video_connector <- ltx2_connector_transformer_1d(
      num_attention_heads = video_connector_num_attention_heads,
      attention_head_dim = video_connector_attention_head_dim,
      num_layers = video_connector_num_layers,
      num_learnable_registers = video_connector_num_learnable_registers,
      rope_base_seq_len = connector_rope_base_seq_len,
      rope_theta = rope_theta,
      rope_double_precision = rope_double_precision,
      causal_temporal_positioning = causal_temporal_positioning,
      rope_type = rope_type
    )

    # Audio connector
    self$audio_connector <- ltx2_connector_transformer_1d(
      num_attention_heads = audio_connector_num_attention_heads,
      attention_head_dim = audio_connector_attention_head_dim,
      num_layers = audio_connector_num_layers,
      num_learnable_registers = audio_connector_num_learnable_registers,
      rope_base_seq_len = connector_rope_base_seq_len,
      rope_theta = rope_theta,
      rope_double_precision = rope_double_precision,
      causal_temporal_positioning = causal_temporal_positioning,
      rope_type = rope_type
    )
  },

  forward = function(
    text_encoder_hidden_states,
    attention_mask,
    additive_mask = FALSE
  ) {
    # Convert to additive attention mask if necessary
    if (!additive_mask) {
      text_dtype <- text_encoder_hidden_states$dtype
      attention_mask <- (attention_mask - 1) $reshape(c(attention_mask$shape[1], 1L, - 1L, attention_mask$shape[length(attention_mask$shape)]))
      attention_mask <- attention_mask$to(dtype = text_dtype) * torch::torch_finfo(text_dtype) $max
    }

    # Project input
    text_encoder_hidden_states <- self$text_proj_in(text_encoder_hidden_states)

    # Video connector
    video_result <- self$video_connector(text_encoder_hidden_states, attention_mask)
    video_text_embedding <- video_result[[1]]
    new_attn_mask <- video_result[[2]]

    # Apply attention mask
    attn_mask <- (new_attn_mask < 1e-6) $to(dtype = torch::torch_int64())
    attn_mask <- attn_mask$reshape(c(video_text_embedding$shape[1], video_text_embedding$shape[2], 1L))
    video_text_embedding <- video_text_embedding * attn_mask
    new_attn_mask <- attn_mask$squeeze(- 1L)

    # Audio connector
    audio_result <- self$audio_connector(text_encoder_hidden_states, attention_mask)
    audio_text_embedding <- audio_result[[1]]

    list(video_text_embedding, audio_text_embedding, new_attn_mask)
  }
)

# -----------------------------------------------------------------------------
# Text Encoder Wrapper
# -----------------------------------------------------------------------------

#' Encode Text for LTX2
#'
#' Encodes text prompts for LTX2 video generation. Supports multiple backends:
#' - "gemma3": Native R torch Gemma3 text encoder
#' - "precomputed": Load pre-computed embeddings from file
#' - "api": Call an HTTP API for text encoding
#' - "random": Generate random embeddings (for testing only)
#'
#' @param prompt Character vector of prompts.
#' @param backend Character. Backend to use ("gemma3", "precomputed", "api", "random").
#' @param model_path Character. Path to Gemma3 model directory (for "gemma3" backend).
#' @param tokenizer_path Character. Path to tokenizer (for "gemma3" backend, defaults to model_path).
#' @param text_encoder Pre-loaded Gemma3 text encoder module (for "gemma3" backend).
#' @param embeddings_file Character. Path to pre-computed embeddings (for "precomputed" backend).
#' @param api_url Character. URL of text encoding API (for "api" backend).
#' @param max_sequence_length Integer. Maximum sequence length (default 1024).
#' @param caption_channels Integer. Caption embedding dimension (default 3840).
#' @param device Character. Device for tensors.
#' @param dtype torch_dtype. Data type for tensors.
#'
#' @return List with prompt_embeds and prompt_attention_mask tensors.
#' @export
encode_text_ltx2 <- function(
  prompt,
  backend = "random",
  model_path = NULL,
  tokenizer_path = NULL,
  text_encoder = NULL,
  embeddings_file = NULL,
  api_url = NULL,
  max_sequence_length = 1024L,
  caption_channels = 3840L,
  device = "cpu",
  dtype = torch::torch_float32()
) {

  if (is.character(prompt) && length(prompt) == 1) {
    prompt <- list(prompt)
  } else {
    prompt <- as.list(prompt)
  }
  batch_size <- length(prompt)

  if (backend == "gemma3") {
    # Native Gemma3 text encoding
    if (identical(dtype, torch::torch_float16())) {
      dtype_str <- "float16"
    } else {
      dtype_str <- "float32"
    }

    result <- encode_with_gemma3(
      prompts = unlist(prompt),
      model = text_encoder %||% model_path,
      tokenizer = tokenizer_path %||% model_path,
      max_sequence_length = max_sequence_length,
      device = device,
      dtype = dtype_str,
      verbose = FALSE
    )

    prompt_embeds <- result$prompt_embeds$to(dtype = dtype)
    prompt_attention_mask <- result$prompt_attention_mask

  } else if (backend == "precomputed") {
    if (is.null(embeddings_file)) {
      stop("embeddings_file required for precomputed backend")
    }
    # Load pre-computed embeddings
    data <- readRDS(embeddings_file)
    prompt_embeds <- torch::torch_tensor(data$embeddings, device = device, dtype = dtype)
    prompt_attention_mask <- torch::torch_tensor(data$attention_mask, device = device, dtype = torch::torch_int64())

  } else if (backend == "api") {
    if (is.null(api_url)) {
      stop("api_url required for api backend")
    }
    # Call HTTP API
    response <- httr::POST(
      api_url,
      body = jsonlite::toJSON(list(
          prompts = prompt,
          max_sequence_length = max_sequence_length
        ), auto_unbox = TRUE),
      httr::content_type_json()
    )
    if (httr::status_code(response) != 200) {
      stop("Text encoding API failed: ", httr::content(response, "text"))
    }
    data <- jsonlite::fromJSON(httr::content(response, "text"))
    prompt_embeds <- torch::torch_tensor(data$embeddings, device = device, dtype = dtype)
    prompt_attention_mask <- torch::torch_tensor(data$attention_mask, device = device, dtype = torch::torch_int64())

  } else if (backend == "random") {
    # Generate random embeddings (for testing)
    # Shape: [batch, seq_len, caption_channels * num_layers] = [B, L, 3840*49]
    # This mimics packed Gemma3 output for testing connectors
    message("Using random embeddings - for testing only")
    packed_dim <- caption_channels * 49L# 49 layers from Gemma3
    prompt_embeds <- torch::torch_randn(c(batch_size, max_sequence_length, packed_dim),
      device = device, dtype = dtype)
    prompt_attention_mask <- torch::torch_ones(c(batch_size, max_sequence_length),
      device = device, dtype = torch::torch_int64())

  } else {
    stop("Unknown backend: ", backend, ". Use 'gemma3', 'precomputed', 'api', or 'random'")
  }

  list(
    prompt_embeds = prompt_embeds,
    prompt_attention_mask = prompt_attention_mask
  )
}

#' Pack Text Embeddings (Gemma-style)
#'
#' Normalizes and packs text encoder hidden states from multiple layers.
#' This is used when working with raw Gemma outputs.
#'
#' @param text_hidden_states Tensor of shape [batch, seq_len, hidden_dim, num_layers].
#' @param sequence_lengths Integer vector of valid sequence lengths per batch item.
#' @param padding_side Character. "left" or "right".
#' @param scale_factor Numeric. Scale factor for normalization (default 8).
#' @param eps Numeric. Epsilon for numerical stability.
#' @param device Character. Device for tensors.
#'
#' @return Tensor of shape [batch, seq_len, hidden_dim * num_layers].
#' @export
pack_text_embeds <- function(
  text_hidden_states,
  sequence_lengths,
  padding_side = "left",
  scale_factor = 8,
  eps = 1e-6,
  device = "cpu"
) {

  dims <- text_hidden_states$shape
  batch_size <- dims[1]
  seq_len <- dims[2]
  hidden_dim <- dims[3]
  num_layers <- dims[4]

  original_dtype <- text_hidden_states$dtype

  # Create padding mask
  token_indices <- torch::torch_arange(start = 0, end = seq_len - 1L, device = device) $unsqueeze(1L)
  sequence_lengths_t <- torch::torch_tensor(sequence_lengths, device = device)

  if (padding_side == "right") {
    mask <- token_indices < sequence_lengths_t$unsqueeze(2L)
  } else if (padding_side == "left") {
    start_indices <- seq_len - sequence_lengths_t$unsqueeze(2L)
    mask <- token_indices >= start_indices
  } else {
    stop("padding_side must be 'left' or 'right'")
  }
  mask <- mask$unsqueeze(- 1L) $unsqueeze(- 1L) # [B, seq_len, 1, 1]

  # Compute masked mean
  masked_states <- text_hidden_states$masked_fill(!mask, 0.0)
  num_valid <- (sequence_lengths_t * hidden_dim) $view(c(batch_size, 1L, 1L, 1L))
  masked_mean <- masked_states$sum(dim = c(2L, 3L), keepdim = TRUE) / (num_valid + eps)

  # Compute min/max
  x_min <- text_hidden_states$masked_fill(!mask, Inf) $amin(dim = c(2L, 3L), keepdim = TRUE)
  x_max <- text_hidden_states$masked_fill(!mask, - Inf) $amax(dim = c(2L, 3L), keepdim = TRUE)

  # Normalize
  normalized <- (text_hidden_states - masked_mean) / (x_max - x_min + eps)
  normalized <- normalized * scale_factor

  # Flatten layers dimension
  normalized <- normalized$flatten(start_dim = 3L)
  mask_flat <- mask$squeeze(- 1L) $expand(c(- 1L, - 1L, hidden_dim * num_layers))
  normalized <- normalized$masked_fill(!mask_flat, 0.0)
  normalized <- normalized$to(dtype = original_dtype)

  normalized
}

# -----------------------------------------------------------------------------
# Weight Loading
# -----------------------------------------------------------------------------

#' Load LTX2 Text Connectors from safetensors
#'
#' Load pre-trained LTX2 connector weights from HuggingFace safetensors file.
#'
#' @param weights_path Character. Path to safetensors file.
#' @param config_path Character. Optional path to config.json.
#' @param device Character. Device to load weights to. Default: "cpu"
#' @param dtype Character. Data type ("float32", "float16"). Default: "float32"
#' @param verbose Logical. Print loading progress. Default: TRUE
#' @return Initialized ltx2_text_connectors module
#' @export
load_ltx2_connectors <- function(
  weights_path,
  config_path = NULL,
  device = "cpu",
  dtype = "float32",
  verbose = TRUE
) {
  if (!file.exists(weights_path)) {
    stop("Weights file not found: ", weights_path)
  }

  # Load config
  config <- NULL
  # Auto-detect config.json in same directory if not specified
  if (is.null(config_path)) {
    auto_config <- file.path(dirname(weights_path), "config.json")
    if (file.exists(auto_config)) {
      config_path <- auto_config
    }
  }
  if (!is.null(config_path) && file.exists(config_path)) {
    config <- jsonlite::fromJSON(config_path)
    if (verbose) message("Loaded config from: ", config_path)
  }

  # Create connectors with config or defaults
  if (!is.null(config)) {
    connectors <- ltx2_text_connectors(
      caption_channels = config$caption_channels %||% 3840L,
      text_proj_in_factor = config$text_proj_in_factor %||% 49L,
      video_connector_num_attention_heads = config$video_connector_num_attention_heads %||% 30L,
      video_connector_attention_head_dim = config$video_connector_attention_head_dim %||% 128L,
      video_connector_num_layers = config$video_connector_num_layers %||% 2L,
      video_connector_num_learnable_registers = as.integer(config$video_connector_num_learnable_registers),
      audio_connector_num_attention_heads = config$audio_connector_num_attention_heads %||% 30L,
      audio_connector_attention_head_dim = config$audio_connector_attention_head_dim %||% 128L,
      audio_connector_num_layers = config$audio_connector_num_layers %||% 2L,
      audio_connector_num_learnable_registers = as.integer(config$audio_connector_num_learnable_registers),
      connector_rope_base_seq_len = config$connector_rope_base_seq_len %||% 4096L,
      rope_theta = config$rope_theta %||% 10000.0,
      rope_double_precision = config$rope_double_precision %||% TRUE,
      causal_temporal_positioning = config$causal_temporal_positioning %||% FALSE,
      rope_type = config$rope_type %||% "split"
    )
  } else {
    connectors <- ltx2_text_connectors()
  }

  # Load weights
  if (verbose) message("Loading weights from: ", weights_path)
  weights <- safetensors::safe_load_file(weights_path, framework = "torch")

  load_ltx2_connector_weights(connectors, weights, verbose = verbose)

  # Move to device
  torch_dtype <- switch(dtype,
    "float32" = torch::torch_float32(),
    "float16" = torch::torch_float16(),
    "bfloat16" = torch::torch_bfloat16(),
    torch::torch_float32()
  )

  connectors$to(device = device, dtype = torch_dtype)

  if (verbose) message("Connectors loaded successfully on device: ", device)
  connectors
}

#' Load weights into LTX2 connectors module
#'
#' @param connectors LTX2 connectors module
#' @param weights Named list of weight tensors
#' @param verbose Print progress
#' @keywords internal
load_ltx2_connector_weights <- function(
  connectors,
  weights,
  verbose = TRUE
) {
  native_params <- names(connectors$parameters)

  remap_connector_key <- function(key) {
    # HuggingFace uses nn.ModuleList for FeedForward:
    #   ff.net.0.proj.weight -> ff.act_fn.proj.weight
    #   ff.net.2.weight -> ff.proj_out.weight
    key <- gsub("\\.ff\\.net\\.0\\.", ".ff.act_fn.", key)
    key <- gsub("\\.ff\\.net\\.2\\.", ".ff.proj_out.", key)

    # to_out.0 is correct - both HF and our module use ModuleList
    key
  }

  loaded <- 0L
  skipped <- 0L
  unmapped <- character(0)

  torch::with_no_grad({
      for (hf_name in names(weights)) {
        native_name <- remap_connector_key(hf_name)

        if (native_name %in% native_params) {
          hf_tensor <- weights[[hf_name]]
          native_tensor <- connectors$parameters[[native_name]]

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
    message(sprintf("Connector weights: %d loaded, %d skipped", loaded, skipped))
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

  invisible(list(loaded = loaded, skipped = skipped, unmapped = unmapped))
}

# Null-coalescing operator (if not already defined)
if (!exists("%||%", mode = "function")) {
  `%||%` <- function(
    x,
    y
  ) if (is.null(x)) y else x
}

