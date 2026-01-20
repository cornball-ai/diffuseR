# LTX2 DiT Transformer Modules
# Audio-Video transformer matching HuggingFace diffusers implementation

# ------------------------------------------------------------------------------
# Helper modules: Timestep embeddings
# ------------------------------------------------------------------------------

#' Get timestep embedding (sinusoidal)
#' @keywords internal
get_timestep_embedding <- function(
  timesteps,
  embedding_dim,
  flip_sin_to_cos = FALSE,
  downscale_freq_shift = 1,
  scale = 1,
  max_period = 10000
) {
  # Ensure timesteps is 1D
  if (timesteps$ndim > 1L) {
    timesteps <- timesteps$flatten()
  }

  # Store original dtype to convert back at end
  orig_dtype <- timesteps$dtype

  half_dim <- embedding_dim %/% 2L

  # Compute in float32 for numerical precision
  exponent <- - log(max_period) * torch::torch_arange(
    start = 0, end = half_dim - 1L, dtype = torch::torch_float32(), device = timesteps$device
  )
  exponent <- exponent / (half_dim - downscale_freq_shift)

  emb <- torch::torch_exp(exponent)
  # timesteps: [N], emb: [half_dim] -> result: [N, half_dim]
  emb <- timesteps$unsqueeze(- 1L) $to(dtype = torch::torch_float32()) * emb$unsqueeze(1L)
  emb <- scale * emb

  # Concat sine and cosine -> [N, embedding_dim]
  emb <- torch::torch_cat(list(torch::torch_sin(emb), torch::torch_cos(emb)), dim = - 1L)

  # Flip if needed (2D tensor: [N, dim])
  if (flip_sin_to_cos) {
    emb <- torch::torch_cat(list(
        emb[, (half_dim + 1L) :embedding_dim],
        emb[, 1L:half_dim]
      ), dim = - 1L)
  }

  # Zero pad if odd
  if (embedding_dim %% 2L == 1L) {
    emb <- torch::nnf_pad(emb, c(0L, 1L, 0L, 0L))
  }

  # Convert back to original dtype (for mixed precision training)
  emb <- emb$to(dtype = orig_dtype)

  emb
}

#' Timesteps module
#' @keywords internal
timesteps_module <- torch::nn_module(
  "Timesteps",
  initialize = function(
    num_channels,
    flip_sin_to_cos = TRUE,
    downscale_freq_shift = 0,
    scale = 1L
  ) {
    self$num_channels <- num_channels
    self$flip_sin_to_cos <- flip_sin_to_cos
    self$downscale_freq_shift <- downscale_freq_shift
    self$scale <- scale
  },
  forward = function(timesteps) {
    get_timestep_embedding(
      timesteps,
      self$num_channels,
      flip_sin_to_cos = self$flip_sin_to_cos,
      downscale_freq_shift = self$downscale_freq_shift,
      scale = self$scale
    )
  }
)

#' Timestep embedding MLP
#' @keywords internal
timestep_embedding_module <- torch::nn_module(
  "TimestepEmbedding",
  initialize = function(
    in_channels,
    time_embed_dim,
    act_fn = "silu",
    out_dim = NULL
  ) {
    self$linear_1 <- make_linear(in_channels, time_embed_dim)

    if (act_fn == "silu") {
      self$act <- torch::nn_silu()
    } else if (act_fn == "gelu") {
      self$act <- torch::nn_gelu()
    } else {
      self$act <- torch::nn_silu()
    }

    if (!is.null(out_dim)) {
      time_embed_dim_out <- out_dim
    } else {
      time_embed_dim_out <- time_embed_dim
    }
    self$linear_2 <- make_linear(time_embed_dim, time_embed_dim_out)
  },
  forward = function(sample) {
    sample <- self$linear_1(sample)
    sample <- self$act(sample)
    sample <- self$linear_2(sample)
    sample
  }
)

#' PixArt Alpha Combined Timestep Size Embeddings
#' @keywords internal
pixart_alpha_combined_timestep_size_embeddings <- torch::nn_module(
  "PixArtAlphaCombinedTimestepSizeEmbeddings",
  initialize = function(
    embedding_dim,
    size_emb_dim,
    use_additional_conditions = FALSE
  ) {
    self$outdim <- size_emb_dim
    self$time_proj <- timesteps_module(num_channels = 256L, flip_sin_to_cos = TRUE, downscale_freq_shift = 0)
    self$timestep_embedder <- timestep_embedding_module(in_channels = 256L, time_embed_dim = embedding_dim)

    self$use_additional_conditions <- use_additional_conditions
    if (use_additional_conditions) {
      self$additional_condition_proj <- timesteps_module(num_channels = 256L, flip_sin_to_cos = TRUE, downscale_freq_shift = 0)
      self$resolution_embedder <- timestep_embedding_module(in_channels = 256L, time_embed_dim = size_emb_dim)
      self$aspect_ratio_embedder <- timestep_embedding_module(in_channels = 256L, time_embed_dim = size_emb_dim)
    }
  },
  forward = function(
    timestep,
    resolution = NULL,
    aspect_ratio = NULL,
    batch_size = NULL,
    hidden_dtype = NULL
  ) {
    timesteps_proj <- self$time_proj(timestep)
    if (!is.null(hidden_dtype)) {
      timesteps_proj <- timesteps_proj$to(dtype = hidden_dtype)
    }
    timesteps_emb <- self$timestep_embedder(timesteps_proj)

    if (self$use_additional_conditions && !is.null(resolution)) {
      resolution_emb <- self$additional_condition_proj(resolution$flatten())
      if (!is.null(hidden_dtype)) {
        resolution_emb <- resolution_emb$to(dtype = hidden_dtype)
      }
      resolution_emb <- self$resolution_embedder(resolution_emb) $reshape(c(batch_size, - 1L))

      aspect_ratio_emb <- self$additional_condition_proj(aspect_ratio$flatten())
      if (!is.null(hidden_dtype)) {
        aspect_ratio_emb <- aspect_ratio_emb$to(dtype = hidden_dtype)
      }
      aspect_ratio_emb <- self$aspect_ratio_embedder(aspect_ratio_emb) $reshape(c(batch_size, - 1L))

      conditioning <- timesteps_emb + torch::torch_cat(list(resolution_emb, aspect_ratio_emb), dim = 2L)
    } else {
      conditioning <- timesteps_emb
    }

    conditioning
  }
)

#' PixArt Alpha Text Projection
#' @keywords internal
pixart_alpha_text_projection <- torch::nn_module(
  "PixArtAlphaTextProjection",
  initialize = function(
    in_features,
    hidden_size,
    out_features = NULL,
    act_fn = "gelu_tanh"
  ) {
    if (is.null(out_features)) out_features <- hidden_size

    self$linear_1 <- make_linear(in_features, hidden_size)

    if (act_fn == "gelu_tanh") {
      self$act_1 <- torch::nn_gelu(approximate = "tanh")
    } else if (act_fn == "silu") {
      self$act_1 <- torch::nn_silu()
    } else {
      self$act_1 <- torch::nn_gelu(approximate = "tanh")
    }

    self$linear_2 <- make_linear(hidden_size, out_features)
  },
  forward = function(caption) {
    hidden_states <- self$linear_1(caption)
    hidden_states <- self$act_1(hidden_states)
    hidden_states <- self$linear_2(hidden_states)
    hidden_states
  }
)

# ------------------------------------------------------------------------------
# FeedForward module
# ------------------------------------------------------------------------------

#' GELU activation with optional approximation
#' @keywords internal
gelu_activation <- torch::nn_module(
  "GELU",
  initialize = function(
    dim_in,
    dim_out,
    approximate = "none",
    bias = TRUE
  ) {
    self$proj <- make_linear(dim_in, dim_out, bias = bias)
    self$approximate <- approximate
  },
  forward = function(hidden_states) {
    hidden_states <- self$proj(hidden_states)
    hidden_states <- torch::nnf_gelu(hidden_states, approximate = self$approximate)
    hidden_states
  }
)

#' FeedForward module
#' @keywords internal
feed_forward <- torch::nn_module(
  "FeedForward",
  initialize = function(
    dim,
    dim_out = NULL,
    mult = 4L,
    dropout = 0.0,
    activation_fn = "gelu-approximate",
    inner_dim = NULL,
    bias = TRUE
  ) {
    if (is.null(inner_dim)) inner_dim <- as.integer(dim * mult)
    if (is.null(dim_out)) dim_out <- dim

    # Activation layer (projects in)
    if (activation_fn == "gelu") {
      self$act_fn <- gelu_activation(dim, inner_dim, approximate = "none", bias = bias)
    } else if (activation_fn == "gelu-approximate") {
      self$act_fn <- gelu_activation(dim, inner_dim, approximate = "tanh", bias = bias)
    } else {
      self$act_fn <- gelu_activation(dim, inner_dim, approximate = "tanh", bias = bias)
    }

    self$dropout <- torch::nn_dropout(p = dropout)
    self$proj_out <- make_linear(inner_dim, dim_out, bias = bias)
  },
  forward = function(hidden_states) {
    hidden_states <- self$act_fn(hidden_states)
    hidden_states <- self$dropout(hidden_states)
    hidden_states <- self$proj_out(hidden_states)
    hidden_states
  }
)

# ------------------------------------------------------------------------------
# RMSNorm
# ------------------------------------------------------------------------------

#' RMS Normalization
#' @keywords internal
rms_norm <- torch::nn_module(
  "RMSNorm",
  initialize = function(
    dim,
    eps = 1e-6,
    elementwise_affine = TRUE
  ) {
    self$eps <- eps
    self$elementwise_affine <- elementwise_affine
    if (elementwise_affine) {
      self$weight <- torch::nn_parameter(torch::torch_ones(dim))
    }
  },
  forward = function(hidden_states) {
    input_dtype <- hidden_states$dtype
    hidden_states <- hidden_states$to(dtype = torch::torch_float32())
    variance <- hidden_states$pow(2) $mean(dim = - 1L, keepdim = TRUE)
    hidden_states <- hidden_states * torch::torch_rsqrt(variance + self$eps)
    if (self$elementwise_affine) {
      hidden_states <- hidden_states * self$weight
    }
    hidden_states$to(dtype = input_dtype)
  }
)

# ------------------------------------------------------------------------------
# LTX2 Adaptive Layer Norm Single
# ------------------------------------------------------------------------------

#' LTX2 AdaLayerNorm Single
#' @keywords internal
ltx2_ada_layer_norm_single <- torch::nn_module(
  "LTX2AdaLayerNormSingle",
  initialize = function(
    embedding_dim,
    num_mod_params = 6L,
    use_additional_conditions = FALSE
  ) {
    self$num_mod_params <- num_mod_params

    self$emb <- pixart_alpha_combined_timestep_size_embeddings(
      embedding_dim = embedding_dim,
      size_emb_dim = embedding_dim %/% 3L,
      use_additional_conditions = use_additional_conditions
    )

    self$silu <- torch::nn_silu()
    self$linear <- make_linear(embedding_dim, num_mod_params * embedding_dim)
  },
  forward = function(
    timestep,
    added_cond_kwargs = NULL,
    batch_size = NULL,
    hidden_dtype = NULL
  ) {
    if (is.null(added_cond_kwargs)) {
      added_cond_kwargs <- list(resolution = NULL, aspect_ratio = NULL)
    }

    embedded_timestep <- self$emb(
      timestep,
      resolution = added_cond_kwargs$resolution,
      aspect_ratio = added_cond_kwargs$aspect_ratio,
      batch_size = batch_size,
      hidden_dtype = hidden_dtype
    )

    mod_params <- self$linear(self$silu(embedded_timestep))
    list(mod_params, embedded_timestep)
  }
)

# ------------------------------------------------------------------------------
# LTX2 Attention
# ------------------------------------------------------------------------------

#' Apply interleaved rotary embedding
#' @keywords internal
apply_interleaved_rotary_emb_list <- function(
  x,
  freqs
) {
  cos_freqs <- freqs[[1]]
  sin_freqs <- freqs[[2]]

  # x: [B, S, C]
  # Split into real and imaginary parts
  x_shape <- x$shape
  x_reshaped <- x$unflatten(3, c(- 1L, 2L)) # [B, S, C//2, 2]
  x_real <- x_reshaped[,,, 1]
  x_imag <- x_reshaped[,,, 2]

  # Rotate: [-x_imag, x_real]
  x_rotated <- torch::torch_stack(list(- x_imag, x_real), dim = - 1L) $flatten(start_dim = 3L)

  # Apply rotation
  out <- (x$to(dtype = torch::torch_float32()) * cos_freqs +
    x_rotated$to(dtype = torch::torch_float32()) * sin_freqs) $to(dtype = x$dtype)

  out
}

#' LTX2 Attention module
#' @keywords internal
ltx2_attention <- torch::nn_module(
  "LTX2Attention",
  initialize = function(
    query_dim,
    heads = 8L,
    kv_heads = 8L,
    dim_head = 64L,
    dropout = 0.0,
    bias = TRUE,
    cross_attention_dim = NULL,
    out_bias = TRUE,
    qk_norm = "rms_norm_across_heads",
    norm_eps = 1e-6,
    norm_elementwise_affine = TRUE,
    rope_type = "interleaved"
  ) {

    self$head_dim <- dim_head
    self$inner_dim <- dim_head * heads
    if (is.null(kv_heads)) {
      self$inner_kv_dim <- self$inner_dim
    } else {
      self$inner_kv_dim <- dim_head * kv_heads
    }
    self$query_dim <- query_dim
    if (!is.null(cross_attention_dim)) {
      self$cross_attention_dim <- cross_attention_dim
    } else {
      self$cross_attention_dim <- query_dim
    }
    self$use_bias <- bias
    self$dropout_p <- dropout
    self$out_dim <- query_dim
    self$heads <- heads
    self$rope_type <- rope_type

    # QK normalization
    self$norm_q <- rms_norm(dim_head * heads, eps = norm_eps, elementwise_affine = norm_elementwise_affine)
    if (is.null(kv_heads)) {
      self$norm_k <- rms_norm(dim_head * heads, eps = norm_eps, elementwise_affine = norm_elementwise_affine)
    } else {
      self$norm_k <- rms_norm(dim_head * kv_heads, eps = norm_eps, elementwise_affine = norm_elementwise_affine)
    }

    # Projections
    self$to_q <- make_linear(query_dim, self$inner_dim, bias = bias)
    self$to_k <- make_linear(self$cross_attention_dim, self$inner_kv_dim, bias = bias)
    self$to_v <- make_linear(self$cross_attention_dim, self$inner_kv_dim, bias = bias)

    # Output projection
    self$to_out <- torch::nn_sequential(
      make_linear(self$inner_dim, self$out_dim, bias = out_bias),
      torch::nn_dropout(p = dropout)
    )
  },
  forward = function(
    hidden_states,
    encoder_hidden_states = NULL,
    attention_mask = NULL,
    query_rotary_emb = NULL,
    key_rotary_emb = NULL
  ) {
    batch_size <- hidden_states$shape[1]

    if (is.null(encoder_hidden_states)) {
      encoder_hidden_states <- hidden_states
    }

    # Project Q, K, V
    query <- self$to_q(hidden_states)
    key <- self$to_k(encoder_hidden_states)
    value <- self$to_v(encoder_hidden_states)

    # Normalize Q, K
    query <- self$norm_q(query)
    key <- self$norm_k(key)

    # Apply RoPE
    if (!is.null(query_rotary_emb)) {
      if (self$rope_type == "interleaved") {
        query <- apply_interleaved_rotary_emb_list(query, query_rotary_emb)
        if (!is.null(key_rotary_emb)) {
          key_rope <- key_rotary_emb
        } else {
          key_rope <- query_rotary_emb
        }
        key <- apply_interleaved_rotary_emb_list(key, key_rope)
      }
    }

    # Reshape for multi-head attention [B, S, H, D]
    query <- query$unflatten(3, c(self$heads, - 1L))
    key <- key$unflatten(3, c(self$heads, - 1L))
    value <- value$unflatten(3, c(self$heads, - 1L))

    # Transpose to [B, H, S, D]
    query <- query$transpose(2L, 3L)
    key <- key$transpose(2L, 3L)
    value <- value$transpose(2L, 3L)

    # Scaled dot-product attention (manual implementation)
    scale <- 1.0 / sqrt(self$head_dim)
    attn_weights <- torch::torch_matmul(query, key$transpose(- 2L, - 1L)) * scale

    if (!is.null(attention_mask)) {
      # Expand attention mask to [B, 1, 1, S] for broadcasting with [B, H, S, S]
      if (attention_mask$ndim == 2L) {
        attention_mask <- attention_mask$unsqueeze(2L) $unsqueeze(2L) # [B, S] -> [B, 1, 1, S]
      } else if (attention_mask$ndim == 3L) {
        attention_mask <- attention_mask$unsqueeze(2L) # [B, 1, S] -> [B, 1, 1, S]
      }
      attn_weights <- attn_weights + attention_mask
    }

    attn_weights <- torch::nnf_softmax(attn_weights, dim = - 1L)

    if (self$training && self$dropout_p > 0) {
      attn_weights <- torch::nnf_dropout(attn_weights, p = self$dropout_p)
    }

    hidden_states <- torch::torch_matmul(attn_weights, value)

    # Reshape back [B, H, S, D] -> [B, S, H*D]
    hidden_states <- hidden_states$transpose(2L, 3L) $flatten(start_dim = 3L)
    hidden_states <- hidden_states$to(dtype = query$dtype)

    # Output projection
    hidden_states <- self$to_out(hidden_states)

    hidden_states
  }
)

# ------------------------------------------------------------------------------
# LTX2 Video Transformer Block
# ------------------------------------------------------------------------------

#' LTX2 Video Transformer Block (Audio-Video)
#' @keywords internal
ltx2_video_transformer_block <- torch::nn_module(
  "LTX2VideoTransformerBlock",
  initialize = function(
    dim,
    num_attention_heads,
    attention_head_dim,
    cross_attention_dim,
    audio_dim,
    audio_num_attention_heads,
    audio_attention_head_dim,
    audio_cross_attention_dim,
    qk_norm = "rms_norm_across_heads",
    activation_fn = "gelu-approximate",
    attention_bias = TRUE,
    attention_out_bias = TRUE,
    eps = 1e-6,
    elementwise_affine = FALSE,
    rope_type = "interleaved"
  ) {

    # 1. Video Self-Attention
    self$norm1 <- rms_norm(dim, eps = eps, elementwise_affine = elementwise_affine)
    self$attn1 <- ltx2_attention(
      query_dim = dim,
      heads = num_attention_heads,
      kv_heads = num_attention_heads,
      dim_head = attention_head_dim,
      bias = attention_bias,
      cross_attention_dim = NULL,
      out_bias = attention_out_bias,
      qk_norm = qk_norm,
      rope_type = rope_type
    )

    # Audio Self-Attention
    self$audio_norm1 <- rms_norm(audio_dim, eps = eps, elementwise_affine = elementwise_affine)
    self$audio_attn1 <- ltx2_attention(
      query_dim = audio_dim,
      heads = audio_num_attention_heads,
      kv_heads = audio_num_attention_heads,
      dim_head = audio_attention_head_dim,
      bias = attention_bias,
      cross_attention_dim = NULL,
      out_bias = attention_out_bias,
      qk_norm = qk_norm,
      rope_type = rope_type
    )

    # 2. Video Cross-Attention (with text)
    self$norm2 <- rms_norm(dim, eps = eps, elementwise_affine = elementwise_affine)
    self$attn2 <- ltx2_attention(
      query_dim = dim,
      cross_attention_dim = cross_attention_dim,
      heads = num_attention_heads,
      kv_heads = num_attention_heads,
      dim_head = attention_head_dim,
      bias = attention_bias,
      out_bias = attention_out_bias,
      qk_norm = qk_norm,
      rope_type = rope_type
    )

    # Audio Cross-Attention (with text)
    self$audio_norm2 <- rms_norm(audio_dim, eps = eps, elementwise_affine = elementwise_affine)
    self$audio_attn2 <- ltx2_attention(
      query_dim = audio_dim,
      cross_attention_dim = audio_cross_attention_dim,
      heads = audio_num_attention_heads,
      kv_heads = audio_num_attention_heads,
      dim_head = audio_attention_head_dim,
      bias = attention_bias,
      out_bias = attention_out_bias,
      qk_norm = qk_norm,
      rope_type = rope_type
    )

    # 3. Audio-to-Video Cross-Attention (Q: Video, K/V: Audio)
    self$audio_to_video_norm <- rms_norm(dim, eps = eps, elementwise_affine = elementwise_affine)
    self$audio_to_video_attn <- ltx2_attention(
      query_dim = dim,
      cross_attention_dim = audio_dim,
      heads = audio_num_attention_heads,
      kv_heads = audio_num_attention_heads,
      dim_head = audio_attention_head_dim,
      bias = attention_bias,
      out_bias = attention_out_bias,
      qk_norm = qk_norm,
      rope_type = rope_type
    )

    # Video-to-Audio Cross-Attention (Q: Audio, K/V: Video)
    self$video_to_audio_norm <- rms_norm(audio_dim, eps = eps, elementwise_affine = elementwise_affine)
    self$video_to_audio_attn <- ltx2_attention(
      query_dim = audio_dim,
      cross_attention_dim = dim,
      heads = audio_num_attention_heads,
      kv_heads = audio_num_attention_heads,
      dim_head = audio_attention_head_dim,
      bias = attention_bias,
      out_bias = attention_out_bias,
      qk_norm = qk_norm,
      rope_type = rope_type
    )

    # 4. Feedforward layers
    self$norm3 <- rms_norm(dim, eps = eps, elementwise_affine = elementwise_affine)
    self$ff <- feed_forward(dim, activation_fn = activation_fn)

    self$audio_norm3 <- rms_norm(audio_dim, eps = eps, elementwise_affine = elementwise_affine)
    self$audio_ff <- feed_forward(audio_dim, activation_fn = activation_fn)

    # 5. Per-layer modulation parameters
    self$scale_shift_table <- torch::nn_parameter(
      torch::torch_randn(6L, dim) / sqrt(dim)
    )
    self$audio_scale_shift_table <- torch::nn_parameter(
      torch::torch_randn(6L, audio_dim) / sqrt(audio_dim)
    )

    # Cross-attention modulation parameters
    self$video_a2v_cross_attn_scale_shift_table <- torch::nn_parameter(torch::torch_randn(5L, dim))
    self$audio_a2v_cross_attn_scale_shift_table <- torch::nn_parameter(torch::torch_randn(5L, audio_dim))
  },
  forward = function(
    hidden_states,
    audio_hidden_states,
    encoder_hidden_states,
    audio_encoder_hidden_states,
    temb,
    temb_audio,
    temb_ca_scale_shift,
    temb_ca_audio_scale_shift,
    temb_ca_gate,
    temb_ca_audio_gate,
    video_rotary_emb = NULL,
    audio_rotary_emb = NULL,
    ca_video_rotary_emb = NULL,
    ca_audio_rotary_emb = NULL,
    encoder_attention_mask = NULL,
    audio_encoder_attention_mask = NULL,
    a2v_cross_attention_mask = NULL,
    v2a_cross_attention_mask = NULL
  ) {

    batch_size <- hidden_states$shape[1]

    # 1. Video and Audio Self-Attention
    norm_hidden_states <- self$norm1(hidden_states)

    # Ada values for video
    num_ada_params <- self$scale_shift_table$shape[1]
    ada_values <- self$scale_shift_table$unsqueeze(1) $unsqueeze(1) $to(device = temb$device, dtype = temb$dtype) +
    temb$reshape(c(batch_size, temb$shape[2], num_ada_params, - 1L))

    shift_msa <- ada_values[,, 1,]
    scale_msa <- ada_values[,, 2,]
    gate_msa <- ada_values[,, 3,]
    shift_mlp <- ada_values[,, 4,]
    scale_mlp <- ada_values[,, 5,]
    gate_mlp <- ada_values[,, 6,]

    norm_hidden_states <- norm_hidden_states * scale_msa$add(1) + shift_msa

    attn_hidden_states <- self$attn1(
      hidden_states = norm_hidden_states,
      encoder_hidden_states = NULL,
      query_rotary_emb = video_rotary_emb
    )
    hidden_states <- hidden_states + attn_hidden_states * gate_msa

    # Audio self-attention
    norm_audio_hidden_states <- self$audio_norm1(audio_hidden_states)

    num_audio_ada_params <- self$audio_scale_shift_table$shape[1]
    audio_ada_values <- self$audio_scale_shift_table$unsqueeze(1) $unsqueeze(1) $to(device = temb_audio$device, dtype = temb_audio$dtype) +
    temb_audio$reshape(c(batch_size, temb_audio$shape[2], num_audio_ada_params, - 1L))

    audio_shift_msa <- audio_ada_values[,, 1,]
    audio_scale_msa <- audio_ada_values[,, 2,]
    audio_gate_msa <- audio_ada_values[,, 3,]
    audio_shift_mlp <- audio_ada_values[,, 4,]
    audio_scale_mlp <- audio_ada_values[,, 5,]
    audio_gate_mlp <- audio_ada_values[,, 6,]

    norm_audio_hidden_states <- norm_audio_hidden_states * audio_scale_msa$add(1) + audio_shift_msa

    attn_audio_hidden_states <- self$audio_attn1(
      hidden_states = norm_audio_hidden_states,
      encoder_hidden_states = NULL,
      query_rotary_emb = audio_rotary_emb
    )
    audio_hidden_states <- audio_hidden_states + attn_audio_hidden_states * audio_gate_msa

    # 2. Video and Audio Cross-Attention with text
    norm_hidden_states <- self$norm2(hidden_states)
    attn_hidden_states <- self$attn2(
      norm_hidden_states,
      encoder_hidden_states = encoder_hidden_states,
      query_rotary_emb = NULL,
      attention_mask = encoder_attention_mask
    )
    hidden_states <- hidden_states + attn_hidden_states

    norm_audio_hidden_states <- self$audio_norm2(audio_hidden_states)
    attn_audio_hidden_states <- self$audio_attn2(
      norm_audio_hidden_states,
      encoder_hidden_states = audio_encoder_hidden_states,
      query_rotary_emb = NULL,
      attention_mask = audio_encoder_attention_mask
    )
    audio_hidden_states <- audio_hidden_states + attn_audio_hidden_states

    # 3. Audio-to-Video and Video-to-Audio Cross-Attention
    norm_hidden_states <- self$audio_to_video_norm(hidden_states)
    norm_audio_hidden_states <- self$video_to_audio_norm(audio_hidden_states)

    # Video cross-attention modulation
    video_per_layer_ca_scale_shift <- self$video_a2v_cross_attn_scale_shift_table[1:4,]
    video_per_layer_ca_gate <- self$video_a2v_cross_attn_scale_shift_table[5:5,]

    video_ca_scale_shift_table <- video_per_layer_ca_scale_shift$unsqueeze(1) $unsqueeze(1) $to(dtype = temb_ca_scale_shift$dtype) +
    temb_ca_scale_shift$reshape(c(batch_size, temb_ca_scale_shift$shape[2], 4L, - 1L))
    video_ca_gate <- video_per_layer_ca_gate$unsqueeze(1) $unsqueeze(1) $to(dtype = temb_ca_gate$dtype) +
    temb_ca_gate$reshape(c(batch_size, temb_ca_gate$shape[2], 1L, - 1L))

    video_a2v_ca_scale <- video_ca_scale_shift_table[,, 1,]
    video_a2v_ca_shift <- video_ca_scale_shift_table[,, 2,]
    video_v2a_ca_scale <- video_ca_scale_shift_table[,, 3,]
    video_v2a_ca_shift <- video_ca_scale_shift_table[,, 4,]
    a2v_gate <- video_ca_gate[,, 1,]

    # Audio cross-attention modulation
    audio_per_layer_ca_scale_shift <- self$audio_a2v_cross_attn_scale_shift_table[1:4,]
    audio_per_layer_ca_gate <- self$audio_a2v_cross_attn_scale_shift_table[5:5,]

    audio_ca_scale_shift_table <- audio_per_layer_ca_scale_shift$unsqueeze(1) $unsqueeze(1) $to(dtype = temb_ca_audio_scale_shift$dtype) +
    temb_ca_audio_scale_shift$reshape(c(batch_size, temb_ca_audio_scale_shift$shape[2], 4L, - 1L))
    audio_ca_gate <- audio_per_layer_ca_gate$unsqueeze(1) $unsqueeze(1) $to(dtype = temb_ca_audio_gate$dtype) +
    temb_ca_audio_gate$reshape(c(batch_size, temb_ca_audio_gate$shape[2], 1L, - 1L))

    audio_a2v_ca_scale <- audio_ca_scale_shift_table[,, 1,]
    audio_a2v_ca_shift <- audio_ca_scale_shift_table[,, 2,]
    audio_v2a_ca_scale <- audio_ca_scale_shift_table[,, 3,]
    audio_v2a_ca_shift <- audio_ca_scale_shift_table[,, 4,]
    v2a_gate <- audio_ca_gate[,, 1,]

    # Audio-to-Video Cross Attention
    mod_norm_hidden_states <- norm_hidden_states * video_a2v_ca_scale$add(1) + video_a2v_ca_shift
    mod_norm_audio_hidden_states <- norm_audio_hidden_states * audio_a2v_ca_scale$add(1) + audio_a2v_ca_shift

    a2v_attn_hidden_states <- self$audio_to_video_attn(
      mod_norm_hidden_states,
      encoder_hidden_states = mod_norm_audio_hidden_states,
      query_rotary_emb = ca_video_rotary_emb,
      key_rotary_emb = ca_audio_rotary_emb,
      attention_mask = a2v_cross_attention_mask
    )
    hidden_states <- hidden_states + a2v_gate * a2v_attn_hidden_states

    # Video-to-Audio Cross Attention
    mod_norm_hidden_states <- norm_hidden_states * video_v2a_ca_scale$add(1) + video_v2a_ca_shift
    mod_norm_audio_hidden_states <- norm_audio_hidden_states * audio_v2a_ca_scale$add(1) + audio_v2a_ca_shift

    v2a_attn_hidden_states <- self$video_to_audio_attn(
      mod_norm_audio_hidden_states,
      encoder_hidden_states = mod_norm_hidden_states,
      query_rotary_emb = ca_audio_rotary_emb,
      key_rotary_emb = ca_video_rotary_emb,
      attention_mask = v2a_cross_attention_mask
    )
    audio_hidden_states <- audio_hidden_states + v2a_gate * v2a_attn_hidden_states

    # 4. Feedforward
    norm_hidden_states <- self$norm3(hidden_states) * scale_mlp$add(1) + shift_mlp
    ff_output <- self$ff(norm_hidden_states)
    hidden_states <- hidden_states + ff_output * gate_mlp

    norm_audio_hidden_states <- self$audio_norm3(audio_hidden_states) * audio_scale_mlp$add(1) + audio_shift_mlp
    audio_ff_output <- self$audio_ff(norm_audio_hidden_states)
    audio_hidden_states <- audio_hidden_states + audio_ff_output * audio_gate_mlp

    list(hidden_states, audio_hidden_states)
  }
)

