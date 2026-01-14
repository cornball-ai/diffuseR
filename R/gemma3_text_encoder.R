# Gemma3 Text Encoder for LTX-2
#
# Native R torch implementation of Gemma3 for text encoding.
# Used by LTX-2 video generation pipeline.
#
# Architecture based on HuggingFace transformers Gemma3TextModel:
# - 48 hidden layers
# - hidden_size: 3840
# - 16 attention heads, 8 KV heads (GQA)
# - Sliding window attention (1024 tokens) on most layers
# - RoPE positional embeddings

# -----------------------------------------------------------------------------
# Gemma3 RMS Normalization
# -----------------------------------------------------------------------------

#' Gemma3 RMS Normalization
#'
#' RMSNorm with optional addition of 1 to weights (Gemma-style).
#'
#' @param dim Integer. Hidden dimension.
#' @param eps Numeric. Epsilon for numerical stability.
#' @keywords internal
gemma3_rms_norm <- torch::nn_module(
  "Gemma3RMSNorm",
  initialize = function(dim, eps = 1e-6) {
    self$eps <- eps
    self$weight <- torch::nn_parameter(torch::torch_zeros(dim))
  },
  forward = function(x) {
    input_dtype <- x$dtype
    x <- x$to(dtype = torch::torch_float32())
    variance <- x$pow(2)$mean(dim = -1L, keepdim = TRUE)
    x <- x * torch::torch_rsqrt(variance + self$eps)
    # Gemma adds 1 to the weight: (1 + weight) * x
    x$to(dtype = input_dtype) * (self$weight$add(1))
  }
)

# -----------------------------------------------------------------------------
# Gemma3 Rotary Position Embeddings
# -----------------------------------------------------------------------------

#' Gemma3 Rotary Position Embeddings
#'
#' Standard RoPE with optional scaling factor for extended context.
#'
#' @param dim Integer. Head dimension.
#' @param max_position_embeddings Integer. Maximum sequence length.
#' @param base Numeric. RoPE base frequency.
#' @param scaling_factor Numeric. Optional scaling factor for extended context.
#' @keywords internal
gemma3_rotary_embedding <- torch::nn_module(
  "Gemma3RotaryEmbedding",
  initialize = function(dim, max_position_embeddings = 8192L, base = 10000.0,
                        scaling_factor = 1.0) {
    self$dim <- dim
    self$max_position_embeddings <- max_position_embeddings
    self$base <- base
    self$scaling_factor <- scaling_factor

    # Precompute inverse frequencies
    inv_freq <- 1.0 / (base ^ (torch::torch_arange(0, dim - 1L, 2L)$to(dtype = torch::torch_float32()) / dim))
    self$inv_freq <- torch::nn_buffer(inv_freq, persistent = FALSE)
  },

  forward = function(x, position_ids) {
    # x: [batch, seq_len, ...]
    # position_ids: [batch, seq_len]

    # Scale positions if using extended context
    position_ids_scaled <- position_ids$to(dtype = torch::torch_float32()) / self$scaling_factor

    # Compute frequencies: [batch, seq_len, dim/2]
    freqs <- torch::torch_einsum("bs,d->bsd", list(position_ids_scaled, self$inv_freq$to(device = x$device)))

    # Duplicate for interleaved: [batch, seq_len, dim]
    emb <- torch::torch_cat(list(freqs, freqs), dim = -1L)

    cos_emb <- torch::torch_cos(emb)$to(dtype = x$dtype)
    sin_emb <- torch::torch_sin(emb)$to(dtype = x$dtype)

    list(cos_emb, sin_emb)
  }
)

#' Apply rotary position embeddings
#'
#' @param q Query tensor [batch, heads, seq, head_dim]
#' @param k Key tensor [batch, heads, seq, head_dim]
#' @param cos Cosine embeddings [batch, seq, head_dim]
#' @param sin Sine embeddings [batch, seq, head_dim]
#' @keywords internal
apply_rotary_pos_emb <- function(q, k, cos, sin) {
  # Reshape cos/sin for broadcasting: [batch, 1, seq, head_dim]
  cos <- cos$unsqueeze(2L)
  sin <- sin$unsqueeze(2L)

  # Apply rotation
  q_embed <- (q * cos) + (rotate_half(q) * sin)
  k_embed <- (k * cos) + (rotate_half(k) * sin)

  list(q_embed, k_embed)
}

#' Rotate half of the hidden dims
#' @keywords internal
rotate_half <- function(x) {
  # x: [..., dim]
  # Split into two halves and rotate
  dim <- x$shape[length(x$shape)]
  half_dim <- dim %/% 2L
  x1 <- x[.., 1:half_dim]
  x2 <- x[.., (half_dim + 1L):dim]
  torch::torch_cat(list(-x2, x1), dim = -1L)
}

# -----------------------------------------------------------------------------
# Gemma3 MLP (Gated Linear Unit)
# -----------------------------------------------------------------------------

#' Gemma3 MLP
#'
#' Feed-forward network with gated linear units and GELU activation.
#'
#' @param config List with hidden_size and intermediate_size.
#' @keywords internal
gemma3_mlp <- torch::nn_module(
  "Gemma3MLP",
  initialize = function(config) {
    self$hidden_size <- config$hidden_size
    self$intermediate_size <- config$intermediate_size

    self$gate_proj <- torch::nn_linear(self$hidden_size, self$intermediate_size, bias = FALSE)
    self$up_proj <- torch::nn_linear(self$hidden_size, self$intermediate_size, bias = FALSE)
    self$down_proj <- torch::nn_linear(self$intermediate_size, self$hidden_size, bias = FALSE)

    # Gemma uses approximate GELU
    self$act_fn <- function(x) torch::nnf_gelu(x, approximate = "tanh")
  },
  forward = function(x) {
    # Gated Linear Unit: down(act(gate(x)) * up(x))
    self$down_proj(self$act_fn(self$gate_proj(x)) * self$up_proj(x))
  }
)

# -----------------------------------------------------------------------------
# Gemma3 Attention
# -----------------------------------------------------------------------------

#' Gemma3 Attention
#'
#' Multi-head attention with Grouped Query Attention (GQA) and optional
#' sliding window attention.
#'
#' @param config Model configuration.
#' @param layer_idx Integer. Layer index for layer-specific settings.
#' @keywords internal
gemma3_attention <- torch::nn_module(
  "Gemma3Attention",
  initialize = function(config, layer_idx = 0L) {
    self$config <- config
    self$layer_idx <- layer_idx

    self$hidden_size <- config$hidden_size
    self$num_heads <- config$num_attention_heads
    self$head_dim <- config$head_dim %||% (config$hidden_size %/% config$num_attention_heads)
    self$num_key_value_heads <- config$num_key_value_heads %||% config$num_attention_heads
    self$num_key_value_groups <- self$num_heads %/% self$num_key_value_heads

    # Sliding window (layer-specific)
    self$sliding_window <- config$sliding_window
    self$is_sliding <- self$get_is_sliding(layer_idx)

    # Attention softcapping (Gemma3 feature)
    self$attn_logit_softcapping <- config$attn_logit_softcapping %||% NULL

    # Projections
    self$q_proj <- torch::nn_linear(self$hidden_size, self$num_heads * self$head_dim, bias = FALSE)
    self$k_proj <- torch::nn_linear(self$hidden_size, self$num_key_value_heads * self$head_dim, bias = FALSE)
    self$v_proj <- torch::nn_linear(self$hidden_size, self$num_key_value_heads * self$head_dim, bias = FALSE)
    self$o_proj <- torch::nn_linear(self$num_heads * self$head_dim, self$hidden_size, bias = FALSE)

    # Q/K normalization (Gemma3 feature)
    self$q_norm <- gemma3_rms_norm(self$head_dim, eps = config$rms_norm_eps %||% 1e-6)
    self$k_norm <- gemma3_rms_norm(self$head_dim, eps = config$rms_norm_eps %||% 1e-6)

    # Scaling factor
    self$scaling <- config$query_pre_attn_scalar %||% self$head_dim
    self$scaling <- 1.0 / sqrt(self$scaling)
  },

  get_is_sliding = function(layer_idx) {
    # Gemma3 uses alternating sliding window pattern
    # Default: every 6th layer is global attention
    sliding_window_pattern <- self$config$sliding_window_pattern %||% 6L
    return((layer_idx + 1L) %% sliding_window_pattern != 0L)
  },

  forward = function(hidden_states, attention_mask = NULL, position_embeddings = NULL) {
    batch_size <- hidden_states$shape[1]
    seq_len <- hidden_states$shape[2]

    # Project Q, K, V
    query_states <- self$q_proj(hidden_states)
    key_states <- self$k_proj(hidden_states)
    value_states <- self$v_proj(hidden_states)

    # Reshape to [batch, heads, seq, head_dim]
    query_states <- query_states$view(c(batch_size, seq_len, self$num_heads, self$head_dim))$transpose(2L, 3L)
    key_states <- key_states$view(c(batch_size, seq_len, self$num_key_value_heads, self$head_dim))$transpose(2L, 3L)
    value_states <- value_states$view(c(batch_size, seq_len, self$num_key_value_heads, self$head_dim))$transpose(2L, 3L)

    # Apply Q/K normalization
    query_states <- self$q_norm(query_states)
    key_states <- self$k_norm(key_states)

    # Apply rotary embeddings
    if (!is.null(position_embeddings)) {
      cos <- position_embeddings[[1]]
      sin <- position_embeddings[[2]]
      rope_result <- apply_rotary_pos_emb(query_states, key_states, cos, sin)
      query_states <- rope_result[[1]]
      key_states <- rope_result[[2]]
    }

    # Repeat K/V for GQA
    if (self$num_key_value_groups > 1L) {
      key_states <- key_states$`repeat`(c(1L, self$num_key_value_groups, 1L, 1L))
      value_states <- value_states$`repeat`(c(1L, self$num_key_value_groups, 1L, 1L))
    }

    # Compute attention scores
    attn_weights <- torch::torch_matmul(query_states, key_states$transpose(-2L, -1L)) * self$scaling

    # Apply softcapping if configured
    if (!is.null(self$attn_logit_softcapping)) {
      attn_weights <- attn_weights / self$attn_logit_softcapping
      attn_weights <- torch::torch_tanh(attn_weights)
      attn_weights <- attn_weights * self$attn_logit_softcapping
    }

    # Apply sliding window mask if needed
    if (self$is_sliding && !is.null(self$sliding_window)) {
      # Create sliding window causal mask
      sliding_mask <- create_sliding_window_mask(seq_len, self$sliding_window, device = attn_weights$device)
      attn_weights <- attn_weights + sliding_mask$unsqueeze(1L)$unsqueeze(1L)
    }

    # Apply attention mask
    if (!is.null(attention_mask)) {
      attn_weights <- attn_weights + attention_mask
    }

    # Softmax
    attn_weights <- torch::nnf_softmax(attn_weights, dim = -1L)
    attn_weights <- attn_weights$to(dtype = value_states$dtype)

    # Apply attention to values
    attn_output <- torch::torch_matmul(attn_weights, value_states)

    # Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
    attn_output <- attn_output$transpose(2L, 3L)$contiguous()
    attn_output <- attn_output$reshape(c(batch_size, seq_len, self$num_heads * self$head_dim))

    # Output projection
    attn_output <- self$o_proj(attn_output)

    attn_output
  }
)

#' Create sliding window causal attention mask
#' @keywords internal
create_sliding_window_mask <- function(seq_len, window_size, device = "cpu") {
  # Create causal mask
  mask <- torch::torch_ones(c(seq_len, seq_len), device = device)
  mask <- torch::torch_triu(mask, diagonal = 1L)

  # Create sliding window mask (tokens beyond window are masked)
  row_idx <- torch::torch_arange(0, seq_len - 1L, device = device)$unsqueeze(2L)
  col_idx <- torch::torch_arange(0, seq_len - 1L, device = device)$unsqueeze(1L)
  window_mask <- (row_idx - col_idx) > window_size

  # Combine: mask future tokens and tokens outside window
  mask <- mask$logical_or(window_mask)

  # Convert to additive mask
  mask <- mask$to(dtype = torch::torch_float32()) * (-1e9)
  mask
}

# -----------------------------------------------------------------------------
# Gemma3 Decoder Layer
# -----------------------------------------------------------------------------

#' Gemma3 Decoder Layer
#'
#' Single transformer block with pre-norm attention and MLP.
#'
#' @param config Model configuration.
#' @param layer_idx Integer. Layer index.
#' @keywords internal
gemma3_decoder_layer <- torch::nn_module(
  "Gemma3DecoderLayer",
  initialize = function(config, layer_idx = 0L) {
    self$hidden_size <- config$hidden_size

    self$self_attn <- gemma3_attention(config, layer_idx = layer_idx)
    self$mlp <- gemma3_mlp(config)

    self$input_layernorm <- gemma3_rms_norm(config$hidden_size, eps = config$rms_norm_eps %||% 1e-6)
    self$post_attention_layernorm <- gemma3_rms_norm(config$hidden_size, eps = config$rms_norm_eps %||% 1e-6)

    # Gemma3 has additional pre-feedforward norm
    self$pre_feedforward_layernorm <- gemma3_rms_norm(config$hidden_size, eps = config$rms_norm_eps %||% 1e-6)
    self$post_feedforward_layernorm <- gemma3_rms_norm(config$hidden_size, eps = config$rms_norm_eps %||% 1e-6)
  },

  forward = function(hidden_states, attention_mask = NULL, position_embeddings = NULL) {
    residual <- hidden_states

    # Pre-norm self-attention
    hidden_states <- self$input_layernorm(hidden_states)
    hidden_states <- self$self_attn(hidden_states,
                                     attention_mask = attention_mask,
                                     position_embeddings = position_embeddings)
    hidden_states <- self$post_attention_layernorm(hidden_states)
    hidden_states <- residual + hidden_states

    # Pre-norm MLP
    residual <- hidden_states
    hidden_states <- self$pre_feedforward_layernorm(hidden_states)
    hidden_states <- self$mlp(hidden_states)
    hidden_states <- self$post_feedforward_layernorm(hidden_states)
    hidden_states <- residual + hidden_states

    hidden_states
  }
)

# -----------------------------------------------------------------------------
# Gemma3 Text Model
# -----------------------------------------------------------------------------

#' Gemma3 Text Model
#'
#' Full Gemma3 text encoder model.
#'
#' @param config Model configuration list.
#' @export
gemma3_text_model <- torch::nn_module(
  "Gemma3TextModel",
  initialize = function(config) {
    self$config <- config
    self$hidden_size <- config$hidden_size
    self$num_layers <- config$num_hidden_layers

    # Token embeddings (scaled by sqrt(hidden_size))
    self$embed_tokens <- torch::nn_embedding(config$vocab_size, config$hidden_size)
    self$embed_scale <- sqrt(config$hidden_size)

    # Rotary embeddings
    head_dim <- config$head_dim %||% (config$hidden_size %/% config$num_attention_heads)
    self$rotary_emb <- gemma3_rotary_embedding(
      dim = head_dim,
      max_position_embeddings = config$max_position_embeddings %||% 8192L,
      base = config$rope_theta %||% 10000.0,
      scaling_factor = config$rope_scaling$factor %||% 1.0
    )

    # Decoder layers
    self$layers <- torch::nn_module_list(lapply(seq_len(self$num_layers), function(i) {
      gemma3_decoder_layer(config, layer_idx = i - 1L)  # 0-indexed
    }))

    # Final norm
    self$norm <- gemma3_rms_norm(config$hidden_size, eps = config$rms_norm_eps %||% 1e-6)
  },

  forward = function(input_ids, attention_mask = NULL, position_ids = NULL,
                     output_hidden_states = FALSE) {
    batch_size <- input_ids$shape[1]
    seq_len <- input_ids$shape[2]

    # Token embeddings (scaled)
    hidden_states <- self$embed_tokens(input_ids) * self$embed_scale

    # Position IDs
    if (is.null(position_ids)) {
      position_ids <- torch::torch_arange(0, seq_len - 1L, device = input_ids$device)
      position_ids <- position_ids$unsqueeze(1L)$expand(c(batch_size, -1L))
    }

    # Compute rotary embeddings
    position_embeddings <- self$rotary_emb(hidden_states, position_ids)

    # Prepare causal attention mask
    if (is.null(attention_mask)) {
      # Create causal mask
      causal_mask <- torch::torch_ones(c(seq_len, seq_len), device = hidden_states$device)
      causal_mask <- torch::torch_triu(causal_mask, diagonal = 1L)
      causal_mask <- causal_mask$to(dtype = hidden_states$dtype) * (-1e9)
      causal_mask <- causal_mask$unsqueeze(1L)$unsqueeze(1L)
    } else {
      # Convert attention mask to additive mask
      # attention_mask: [batch, seq_len] with 1 for valid, 0 for padding
      causal_mask <- torch::torch_ones(c(seq_len, seq_len), device = hidden_states$device)
      causal_mask <- torch::torch_triu(causal_mask, diagonal = 1L)

      # Expand padding mask: [batch, 1, 1, seq_len]
      padding_mask <- (1 - attention_mask)$unsqueeze(2L)$unsqueeze(2L)

      # Combine masks
      causal_mask <- (causal_mask$unsqueeze(1L)$unsqueeze(1L) + padding_mask)$to(dtype = hidden_states$dtype) * (-1e9)
    }

    # Collect hidden states if requested
    all_hidden_states <- if (output_hidden_states) list(hidden_states) else NULL

    # Apply decoder layers
    for (i in seq_along(self$layers)) {
      layer <- self$layers[[i]]
      hidden_states <- layer(hidden_states,
                              attention_mask = causal_mask,
                              position_embeddings = position_embeddings)

      if (output_hidden_states) {
        all_hidden_states <- c(all_hidden_states, list(hidden_states))
      }
    }

    # Final normalization
    hidden_states <- self$norm(hidden_states)

    if (output_hidden_states) {
      all_hidden_states <- c(all_hidden_states, list(hidden_states))
    }

    list(
      last_hidden_state = hidden_states,
      hidden_states = all_hidden_states
    )
  }
)

# -----------------------------------------------------------------------------
# Gemma3 Config
# -----------------------------------------------------------------------------

#' Create Gemma3 configuration for LTX-2
#'
#' Returns the default configuration used by LTX-2's text encoder.
#'
#' @return List with model configuration parameters.
#' @export
gemma3_config_ltx2 <- function() {
  list(
    vocab_size = 262208L,
    hidden_size = 3840L,
    intermediate_size = 15360L,
    num_hidden_layers = 48L,
    num_attention_heads = 16L,
    num_key_value_heads = 8L,
    head_dim = 256L,
    max_position_embeddings = 131072L,
    rms_norm_eps = 1e-6,
    rope_theta = 10000.0,
    rope_scaling = list(factor = 8.0, type = "linear"),
    sliding_window = 1024L,
    sliding_window_pattern = 6L,
    attn_logit_softcapping = 50.0,
    query_pre_attn_scalar = 256L
  )
}

# -----------------------------------------------------------------------------
# Weight Loading
# -----------------------------------------------------------------------------

#' Load Gemma3 Text Model from safetensors
#'
#' Loads pre-trained Gemma3 weights from HuggingFace safetensors files.
#'
#' @param model_path Character. Path to directory containing model files.
#' @param device Character. Device to load model to.
#' @param dtype Character. Data type ("float32", "float16", "bfloat16").
#' @param verbose Logical. Print loading progress.
#' @return Initialized gemma3_text_model with loaded weights.
#' @export
load_gemma3_text_encoder <- function(model_path, device = "cpu",
                                      dtype = "float16", verbose = TRUE) {
  # Load config
  config_path <- file.path(model_path, "config.json")
  if (!file.exists(config_path)) {
    stop("Config file not found: ", config_path)
  }

  config_raw <- jsonlite::fromJSON(config_path)

  # Extract text config if this is a multimodal model
  if (!is.null(config_raw$text_config)) {
    config_raw <- config_raw$text_config
  }

  config <- list(
    vocab_size = config_raw$vocab_size %||% 262208L,
    hidden_size = config_raw$hidden_size %||% 3840L,
    intermediate_size = config_raw$intermediate_size %||% 15360L,
    num_hidden_layers = config_raw$num_hidden_layers %||% 48L,
    num_attention_heads = config_raw$num_attention_heads %||% 16L,
    num_key_value_heads = config_raw$num_key_value_heads %||% 8L,
    head_dim = config_raw$head_dim %||% 256L,
    max_position_embeddings = config_raw$max_position_embeddings %||% 131072L,
    rms_norm_eps = config_raw$rms_norm_eps %||% 1e-6,
    rope_theta = config_raw$rope_theta %||% 10000.0,
    rope_scaling = list(factor = config_raw$rope_scaling$factor %||% 8.0),
    sliding_window = config_raw$sliding_window %||% 1024L,
    sliding_window_pattern = config_raw$sliding_window_pattern %||% 6L,
    attn_logit_softcapping = config_raw$attn_logit_softcapping %||% 50.0,
    query_pre_attn_scalar = config_raw$query_pre_attn_scalar %||% 256L
  )

  if (verbose) {
    message(sprintf("Creating Gemma3 model: %d layers, hidden_size=%d",
                    config$num_hidden_layers, config$hidden_size))
  }

  # Create model
  model <- gemma3_text_model(config)

  # Find safetensor files
  safetensor_files <- list.files(model_path, pattern = "\\.safetensors$", full.names = TRUE)
  # Prefer model-* files over diffusion_pytorch_model-*
  model_files <- grep("^model-", basename(safetensor_files), value = TRUE)
  if (length(model_files) == 0) {
    model_files <- grep("diffusion_pytorch_model", safetensor_files, value = TRUE)
  } else {
    model_files <- file.path(model_path, model_files)
    safetensor_files <- model_files
  }

  if (length(safetensor_files) == 0) {
    stop("No safetensor files found in: ", model_path)
  }

  if (verbose) {
    message(sprintf("Loading weights from %d safetensor files...", length(safetensor_files)))
  }

  # Load and apply weights
  total_loaded <- 0L
  total_skipped <- 0L

  for (sf_path in safetensor_files) {
    if (verbose) message("  Loading: ", basename(sf_path))
    weights <- safetensors::safe_load_file(sf_path, framework = "torch")

    result <- load_gemma3_weights(model, weights, verbose = FALSE)
    total_loaded <- total_loaded + result$loaded
    total_skipped <- total_skipped + result$skipped
  }

  if (verbose) {
    message(sprintf("Loaded %d parameters, skipped %d", total_loaded, total_skipped))
  }

  # Move to device with dtype
  torch_dtype <- switch(dtype,
    "float32" = torch::torch_float32(),
    "float16" = torch::torch_float16(),
    "bfloat16" = torch::torch_bfloat16(),
    torch::torch_float32()
  )

  model$to(device = device, dtype = torch_dtype)

  if (verbose) message("Gemma3 text encoder loaded on device: ", device)

  model
}

#' Load weights into Gemma3 model
#' @keywords internal
load_gemma3_weights <- function(model, weights, verbose = TRUE) {
  native_params <- names(model$parameters)

  # Remap HuggingFace keys to our module structure
  remap_gemma3_key <- function(key) {
    # Remove 'language_model.' prefix if present (for multimodal models)
    key <- sub("^language_model\\.", "", key)

    # Map HuggingFace structure to our structure:
    # model.embed_tokens -> embed_tokens
    # model.layers.0.self_attn.q_proj -> layers.1.self_attn.q_proj (1-indexed)
    # model.norm -> norm

    key <- sub("^model\\.", "", key)

    # Convert 0-indexed layer numbers to 1-indexed
    # Use gregexpr to find and replace layer numbers
    matches <- gregexpr("layers\\.(\\d+)\\.", key)
    if (matches[[1]][1] != -1) {
      # Extract the layer number
      layer_match <- regmatches(key, matches)[[1]]
      layer_num <- as.integer(gsub("layers\\.(\\d+)\\.", "\\1", layer_match)) + 1L
      key <- sub("layers\\.\\d+\\.", paste0("layers.", layer_num, "."), key)
    }

    key
  }

  loaded <- 0L
  skipped <- 0L

  torch::with_no_grad({
    for (hf_name in names(weights)) {
      native_name <- remap_gemma3_key(hf_name)

      if (native_name %in% native_params) {
        hf_tensor <- weights[[hf_name]]
        native_tensor <- model$parameters[[native_name]]

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
      }
    }
  })

  if (verbose) {
    message(sprintf("Gemma3 weights: %d loaded, %d skipped", loaded, skipped))
  }

  invisible(list(loaded = loaded, skipped = skipped))
}

# -----------------------------------------------------------------------------
# Tokenizer Wrapper
# -----------------------------------------------------------------------------

#' Gemma3 Tokenizer
#'
#' Native R tokenizer for Gemma3 using BPE.
#' Loads from HuggingFace tokenizer.json format.
#'
#' @param tokenizer_path Character. Path to tokenizer directory or tokenizer.json file.
#' @return A gemma3_tokenizer object (extends bpe_tokenizer).
#' @export
gemma3_tokenizer <- function(tokenizer_path) {
  # Load base BPE tokenizer
  tokenizer <- bpe_tokenizer(tokenizer_path)

  # Add Gemma-specific configuration
  tokenizer$padding_side <- "left"  # Gemma uses left padding

  # Update class

  class(tokenizer) <- c("gemma3_tokenizer", class(tokenizer))

  tokenizer
}

#' Tokenize text for Gemma3
#'
#' @param tokenizer Gemma3 tokenizer object.
#' @param text Character vector of prompts.
#' @param max_length Integer. Maximum sequence length.
#' @param padding Character. Padding strategy ("left", "right", "max_length", "none").
#' @param return_tensors Character. Return type ("list" or "pt" for torch tensors).
#' @return List with input_ids and attention_mask.
#' @export
tokenize_gemma3 <- function(tokenizer, text, max_length = 1024L,
                             padding = "max_length", return_tensors = "pt") {
  if (!inherits(tokenizer, "gemma3_tokenizer") && !inherits(tokenizer, "bpe_tokenizer")) {
    stop("tokenizer must be a gemma3_tokenizer or bpe_tokenizer object")
  }

  # Use native BPE encoding
  result <- encode_bpe(
    tokenizer = tokenizer,
    text = text,
    add_special_tokens = TRUE,
    max_length = max_length,
    padding = padding,
    truncation = TRUE,
    return_tensors = return_tensors
  )

  result
}

# -----------------------------------------------------------------------------
# Full Text Encoding Pipeline
# -----------------------------------------------------------------------------

#' Encode text with Gemma3 for LTX-2
#'
#' Full pipeline for encoding text prompts using Gemma3 text encoder.
#' Returns packed embeddings ready for LTX-2 connectors.
#'
#' @param prompts Character vector of prompts.
#' @param model Gemma3 text model (or path to load from).
#' @param tokenizer Gemma3 tokenizer (or path to load from).
#' @param max_sequence_length Integer. Maximum sequence length.
#' @param scale_factor Numeric. Scale factor for packing (default 8).
#' @param device Character. Device for computation.
#' @param dtype Character. Data type.
#' @param verbose Logical. Print progress.
#' @return List with prompt_embeds and prompt_attention_mask.
#' @export
encode_with_gemma3 <- function(prompts, model = NULL, tokenizer = NULL,
                                max_sequence_length = 1024L,
                                scale_factor = 8,
                                device = "cuda",
                                dtype = "float16",
                                verbose = TRUE) {
  # Load model if path provided
  if (is.character(model)) {
    model <- load_gemma3_text_encoder(model, device = device, dtype = dtype, verbose = verbose)
  }

  # Load tokenizer if path provided
  if (is.character(tokenizer)) {
    tokenizer <- gemma3_tokenizer(tokenizer)
  }

  if (is.null(model) || is.null(tokenizer)) {
    stop("Both model and tokenizer are required")
  }

  # Ensure prompts is a list
  if (is.character(prompts)) {
    prompts <- as.list(prompts)
  }

  # Tokenize
  if (verbose) message("Tokenizing prompts...")
  tokens <- tokenize_gemma3(tokenizer, unlist(prompts),
                             max_length = max_sequence_length,
                             padding = "left")

  input_ids <- tokens$input_ids$to(device = device)
  attention_mask <- tokens$attention_mask$to(device = device)

  # Run through model
  if (verbose) message("Encoding with Gemma3...")
  torch::with_no_grad({
    output <- model(input_ids, attention_mask = attention_mask, output_hidden_states = TRUE)
  })

  # Stack hidden states from all layers
  hidden_states_list <- output$hidden_states
  # Stack: [batch, seq_len, hidden_size, num_layers+1]
  hidden_states_stacked <- torch::torch_stack(hidden_states_list, dim = -1L)

  # Compute sequence lengths from attention mask
  sequence_lengths <- as.integer(attention_mask$sum(dim = 2L)$cpu())

  # Pack embeddings
  if (verbose) message("Packing embeddings...")
  prompt_embeds <- pack_text_embeds(
    hidden_states_stacked,
    sequence_lengths = sequence_lengths,
    padding_side = "left",
    scale_factor = scale_factor,
    device = device
  )

  list(
    prompt_embeds = prompt_embeds,
    prompt_attention_mask = attention_mask
  )
}

# Null-coalescing operator
if (!exists("%||%", mode = "function")) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
}
