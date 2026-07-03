#' LTX-2.3 Text Embedding Connectors
#'
#' Fresh R port of the LTX text connectors from the diffusers reference
#' (Apache-2.0, src/diffusers/pipelines/ltx2/connectors.py). The
#' connectors take raw stacked per-layer Gemma3 hidden states
#' [batch, seq, caption_channels, num_layers + 1], normalize and project
#' them per modality, replace padding with learnable registers, and run a
#' small 1D transformer per modality to produce the DiT text embeddings.
#'
#' @name connectors_ltx23
NULL

#' Per-token RMS normalization over the channel axis
#'
#' @param x Tensor [B, S, C, L] of stacked per-layer hidden states.
#' @param eps Numeric. Stability epsilon.
#'
#' @return Tensor of the same shape.
#'
#' @export
ltx23_per_token_rms_norm <- function(x, eps = 1e-6) {
  variance <- torch::torch_mean(x^2, dim = 3L, keepdim = TRUE)
  x * torch::torch_rsqrt(variance + eps)
}

#' 1D rotary embeddings for the text connectors
#'
#' @param dim Integer. Rotary dimension (connector inner dim).
#' @param base_seq_len Integer. Base sequence length for normalization.
#' @param theta Numeric. RoPE theta.
#' @param double_precision Logical. Compute base frequencies in float64.
#' @param rope_type "split" (LTX-2.3) or "interleaved".
#' @param num_attention_heads Integer. For the split per-head layout.
#'
#' @export
ltx23_rotary_pos_embed_1d <- torch::nn_module(
  "ltx23_rotary_pos_embed_1d",
  initialize = function(
    dim,
    base_seq_len = 4096L,
    theta = 10000.0,
    double_precision = TRUE,
    rope_type = "split",
    num_attention_heads = 32L
  ) {
    if (!rope_type %in% c("interleaved", "split")) {
      stop("rope_type must be 'interleaved' or 'split', got: ", rope_type)
    }
    self$dim <- as.integer(dim)
    self$base_seq_len <- base_seq_len
    self$theta <- theta
    self$double_precision <- double_precision
    self$rope_type <- rope_type
    self$num_attention_heads <- as.integer(num_attention_heads)
  },
  forward = function(batch_size, pos, device) {
    grid_1d <- torch::torch_arange(
      start = 0, end = pos - 1,
      dtype = torch::torch_float32(), device = device
    ) / self$base_seq_len
    grid <- grid_1d$unsqueeze(1L)$`repeat`(c(batch_size, 1L))# [B, S]

    freqs_dtype <- if (self$double_precision) torch::torch_float64() else torch::torch_float32()
    pow_indices <- torch::torch_pow(
      self$theta,
      torch::torch_linspace(
        start = 0.0, end = 1.0, steps = self$dim %/% 2L,
        dtype = freqs_dtype, device = device
      )
    )
    freqs <- (pow_indices * pi / 2.0)$to(dtype = torch::torch_float32())
    freqs <- (grid$unsqueeze(-1L) * 2 - 1) * freqs# [B, S, dim/2]

    if (self$rope_type == "interleaved") {
      cos_freqs <- freqs$cos()$repeat_interleave(2L, dim = -1L)
      sin_freqs <- freqs$sin()$repeat_interleave(2L, dim = -1L)
    } else {
      cos_freqs <- freqs$cos()
      sin_freqs <- freqs$sin()
      b <- cos_freqs$shape[1]
      t <- cos_freqs$shape[2]
      cos_freqs <- cos_freqs$reshape(c(b, t, self$num_attention_heads, -1L))$transpose(2L, 3L)
      sin_freqs <- sin_freqs$reshape(c(b, t, self$num_attention_heads, -1L))$transpose(2L, 3L)
    }
    list(cos_freqs, sin_freqs)
  }
)

# Pre-norm 1D transformer block: RMS norm -> attention -> RMS norm -> FF,
# both with plain residual connections.
ltx23_transformer_block_1d <- torch::nn_module(
  "ltx23_transformer_block_1d",
  initialize = function(
    dim,
    num_attention_heads,
    attention_head_dim,
    eps = 1e-6,
    rope_type = "split",
    apply_gated_attention = TRUE
  ) {
    self$norm1 <- ltx23_rms_norm(dim, eps = eps, elementwise_affine = FALSE)
    self$attn1 <- ltx23_attention(
      query_dim = dim, heads = num_attention_heads, dim_head = attention_head_dim,
      rope_type = rope_type, apply_gated_attention = apply_gated_attention
    )
    self$norm2 <- ltx23_rms_norm(dim, eps = eps, elementwise_affine = FALSE)
    self$ff <- ltx23_feed_forward(dim)
  },
  forward = function(hidden_states, attention_mask = NULL, rotary_emb = NULL) {
    norm_hidden_states <- self$norm1(hidden_states)
    hidden_states <- hidden_states + self$attn1(
      norm_hidden_states, attention_mask = attention_mask, query_rotary_emb = rotary_emb
    )
    norm_hidden_states <- self$norm2(hidden_states)
    hidden_states + self$ff(norm_hidden_states)
  }
)

#' 1D connector transformer
#'
#' Replaces padded positions with learnable registers (valid tokens are
#' front-aligned in their original order; the tail is filled with
#' registers indexed by absolute position, after which the attention mask
#' is cleared), then runs 1D transformer blocks with rotary embeddings.
#'
#' @param num_attention_heads,attention_head_dim,num_layers Transformer shape.
#' @param num_learnable_registers Integer or NULL. Register count (the
#'   sequence length must be divisible by it).
#' @param rope_base_seq_len,rope_theta,rope_double_precision,rope_type RoPE config.
#' @param eps Numeric. Norm epsilon.
#' @param gated_attention Logical. Per-head attention output gates.
#'
#' @export
ltx23_connector_transformer_1d <- torch::nn_module(
  "ltx23_connector_transformer_1d",
  initialize = function(
    num_attention_heads = 32L,
    attention_head_dim = 128L,
    num_layers = 8L,
    num_learnable_registers = 128L,
    rope_base_seq_len = 4096L,
    rope_theta = 10000.0,
    rope_double_precision = TRUE,
    eps = 1e-6,
    rope_type = "split",
    gated_attention = TRUE
  ) {
    self$num_attention_heads <- as.integer(num_attention_heads)
    self$inner_dim <- as.integer(num_attention_heads * attention_head_dim)
    self$num_learnable_registers <- num_learnable_registers

    if (!is.null(num_learnable_registers)) {
      self$learnable_registers <- torch::nn_parameter(
        torch::torch_rand(num_learnable_registers, self$inner_dim) * 2 - 1
      )
    }

    self$rope <- ltx23_rotary_pos_embed_1d(
      self$inner_dim,
      base_seq_len = rope_base_seq_len,
      theta = rope_theta,
      double_precision = rope_double_precision,
      rope_type = rope_type,
      num_attention_heads = num_attention_heads
    )

    self$transformer_blocks <- torch::nn_module_list(lapply(seq_len(num_layers), function(i) {
      ltx23_transformer_block_1d(
        dim = self$inner_dim,
        num_attention_heads = num_attention_heads,
        attention_head_dim = attention_head_dim,
        eps = eps,
        rope_type = rope_type,
        apply_gated_attention = gated_attention
      )
    }))

    self$norm_out <- ltx23_rms_norm(self$inner_dim, eps = eps, elementwise_affine = FALSE)
  },
  forward = function(hidden_states, attention_mask = NULL,
    attn_mask_binarize_threshold = -9000.0) {
    batch_size <- hidden_states$shape[1]
    seq_len <- hidden_states$shape[2]

    if (!is.null(self$num_learnable_registers)) {
      if (seq_len %% self$num_learnable_registers != 0L) {
        stop(
          "Sequence length ", seq_len, " must be divisible by the number of ",
          "learnable registers ", self$num_learnable_registers
        )
      }
      num_repeats <- seq_len %/% self$num_learnable_registers
      registers <- self$learnable_registers$unsqueeze(1L)$
        expand(c(num_repeats, -1L, -1L))$reshape(c(seq_len, -1L))# [S, D]

      binary_attn_mask <- (attention_mask >= attn_mask_binarize_threshold)$to(
        dtype = torch::torch_int()
      )
      if (binary_attn_mask$ndim == 4L) {
        binary_attn_mask <- binary_attn_mask$squeeze(2L)$squeeze(2L)# [B, 1, 1, S] -> [B, S]
      }

      # Front-align valid tokens (stable sort keeps their order), fill the
      # tail with registers indexed by absolute position
      order <- torch::torch_sort(1L - binary_attn_mask, dim = 2L, stable = TRUE)[[2]]# [B, S]
      front_aligned <- torch::torch_gather(
        hidden_states, 2L,
        order$unsqueeze(-1L)$expand(c(-1L, -1L, hidden_states$shape[3]))
      )
      num_valid <- binary_attn_mask$sum(dim = 2L, keepdim = TRUE)# [B, 1]
      positions <- torch::torch_arange(
        start = 0, end = seq_len - 1, device = hidden_states$device
      )$unsqueeze(1L)# [1, S]
      front_mask <- (positions < num_valid)$unsqueeze(-1L)# [B, S, 1]
      registers_expanded <- registers$unsqueeze(1L)$expand(c(batch_size, -1L, -1L))
      hidden_states <- torch::torch_where(
        front_mask, front_aligned, registers_expanded$to(dtype = hidden_states$dtype)
      )

      # All positions are valid once registers replace the padding
      attention_mask <- torch::torch_zeros_like(attention_mask)
    }

    rotary_emb <- self$rope(batch_size, seq_len, device = hidden_states$device)

    for (i in seq_along(self$transformer_blocks)) {
      hidden_states <- self$transformer_blocks[[i]](
        hidden_states, attention_mask = attention_mask, rotary_emb = rotary_emb
      )
    }
    hidden_states <- self$norm_out(hidden_states)

    list(hidden_states, attention_mask)
  }
)

#' LTX-2.3 text connectors
#'
#' Takes raw stacked per-layer text encoder hidden states and produces
#' the video and audio text embeddings for the DiT: per-token RMS norm,
#' per-modality sqrt(dim ratio) rescaling and projection, then a
#' per-modality 1D connector transformer.
#'
#' @param caption_channels Integer. Text encoder hidden size (3840 for
#'   Gemma3-12B).
#' @param text_proj_in_factor Integer. Number of stacked hidden states
#'   (num_layers + 1 = 49 for Gemma3-12B).
#' @param video_connector_num_attention_heads,video_connector_attention_head_dim,video_connector_num_layers
#'   Video connector shape (LTX-2.3: 32 x 128, 8 layers).
#' @param video_connector_num_learnable_registers Integer or NULL.
#' @param video_gated_attn Logical.
#' @param audio_connector_num_attention_heads,audio_connector_attention_head_dim,audio_connector_num_layers
#'   Audio connector shape (LTX-2.3: 32 x 64, 8 layers).
#' @param audio_connector_num_learnable_registers Integer or NULL.
#' @param audio_gated_attn Logical.
#' @param connector_rope_base_seq_len,rope_theta,rope_double_precision,rope_type RoPE config.
#' @param video_hidden_dim,audio_hidden_dim Integers. Projection targets
#'   (DiT inner dims: 4096 / 2048).
#' @param proj_bias Logical. Projection bias (TRUE for LTX-2.3).
#'
#' @export
ltx23_text_connectors <- torch::nn_module(
  "ltx23_text_connectors",
  initialize = function(
    caption_channels = 3840L,
    text_proj_in_factor = 49L,
    video_connector_num_attention_heads = 32L,
    video_connector_attention_head_dim = 128L,
    video_connector_num_layers = 8L,
    video_connector_num_learnable_registers = 128L,
    video_gated_attn = TRUE,
    audio_connector_num_attention_heads = 32L,
    audio_connector_attention_head_dim = 64L,
    audio_connector_num_layers = 8L,
    audio_connector_num_learnable_registers = 128L,
    audio_gated_attn = TRUE,
    connector_rope_base_seq_len = 4096L,
    rope_theta = 10000.0,
    rope_double_precision = TRUE,
    rope_type = "split",
    video_hidden_dim = 4096L,
    audio_hidden_dim = 2048L,
    proj_bias = TRUE
  ) {
    self$caption_channels <- as.integer(caption_channels)
    self$video_hidden_dim <- as.integer(video_hidden_dim)
    self$audio_hidden_dim <- as.integer(audio_hidden_dim)

    text_encoder_dim <- caption_channels * text_proj_in_factor
    self$video_text_proj_in <- torch::nn_linear(text_encoder_dim, video_hidden_dim, bias = proj_bias)
    self$audio_text_proj_in <- torch::nn_linear(text_encoder_dim, audio_hidden_dim, bias = proj_bias)

    self$video_connector <- ltx23_connector_transformer_1d(
      num_attention_heads = video_connector_num_attention_heads,
      attention_head_dim = video_connector_attention_head_dim,
      num_layers = video_connector_num_layers,
      num_learnable_registers = video_connector_num_learnable_registers,
      rope_base_seq_len = connector_rope_base_seq_len,
      rope_theta = rope_theta,
      rope_double_precision = rope_double_precision,
      rope_type = rope_type,
      gated_attention = video_gated_attn
    )
    self$audio_connector <- ltx23_connector_transformer_1d(
      num_attention_heads = audio_connector_num_attention_heads,
      attention_head_dim = audio_connector_attention_head_dim,
      num_layers = audio_connector_num_layers,
      num_learnable_registers = audio_connector_num_learnable_registers,
      rope_base_seq_len = connector_rope_base_seq_len,
      rope_theta = rope_theta,
      rope_double_precision = rope_double_precision,
      rope_type = rope_type,
      gated_attention = audio_gated_attn
    )
  },
  forward = function(text_encoder_hidden_states, attention_mask) {
    if (text_encoder_hidden_states$ndim == 3L) {
      text_encoder_hidden_states <- text_encoder_hidden_states$unflatten(
        3L, c(self$caption_channels, -1L)
      )
    }

    norm_states <- ltx23_per_token_rms_norm(text_encoder_hidden_states)
    norm_states <- norm_states$flatten(start_dim = 3L, end_dim = 4L)
    bool_mask <- attention_mask$to(dtype = torch::torch_bool())$unsqueeze(-1L)
    norm_states <- torch::torch_where(
      bool_mask, norm_states, torch::torch_zeros_like(norm_states)
    )

    # Rescale per modality by sqrt(target_dim / caption_channels)
    video_norm <- norm_states$mul(sqrt(self$video_hidden_dim / self$caption_channels))
    audio_norm <- norm_states$mul(sqrt(self$audio_hidden_dim / self$caption_channels))

    video_proj <- self$video_text_proj_in(video_norm)
    audio_proj <- self$audio_text_proj_in(audio_norm)

    # Multiplicative [B, S] mask -> additive [B, 1, 1, S] with -finfo max
    text_dtype <- video_proj$dtype
    add_mask <- (attention_mask$to(dtype = torch::torch_int64()) - 1L)$to(dtype = text_dtype)
    add_mask <- add_mask$reshape(c(add_mask$shape[1], 1L, -1L, add_mask$shape[2]))
    add_mask <- add_mask * torch::torch_finfo(text_dtype)$max

    video_res <- self$video_connector(video_proj, add_mask)
    video_text_embedding <- video_res[[1]]
    video_attn_mask <- video_res[[2]]

    # Post-connector mask back to binary; mask the video embedding
    binary_mask <- (video_attn_mask < 1e-6)$to(dtype = torch::torch_int64())
    binary_mask <- binary_mask$reshape(c(
      video_text_embedding$shape[1], video_text_embedding$shape[2], 1L
    ))
    video_text_embedding <- video_text_embedding * binary_mask

    audio_res <- self$audio_connector(audio_proj, add_mask)
    audio_text_embedding <- audio_res[[1]]

    list(
      video_text_embedding = video_text_embedding,
      audio_text_embedding = audio_text_embedding,
      attention_mask = binary_mask$squeeze(-1L)
    )
  }
)

#' Map an official connectors checkpoint key to the R module name
#'
#' @param key Character. Checkpoint key.
#'
#' @return Character. Module parameter name.
#'
#' @export
ltx23_map_connector_key <- function(key) {
  key <- sub("^model\\.diffusion_model\\.", "", key)
  key <- gsub("video_embeddings_connector", "video_connector", key, fixed = TRUE)
  key <- gsub("audio_embeddings_connector", "audio_connector", key, fixed = TRUE)
  key <- gsub("transformer_1d_blocks", "transformer_blocks", key, fixed = TRUE)
  key <- gsub("text_embedding_projection.video_aggregate_embed", "video_text_proj_in", key, fixed = TRUE)
  key <- gsub("text_embedding_projection.audio_aggregate_embed", "audio_text_proj_in", key, fixed = TRUE)
  key <- gsub("q_norm", "norm_q", key, fixed = TRUE)
  key <- gsub("k_norm", "norm_k", key, fixed = TRUE)
  key
}
