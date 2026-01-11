#' QuickGELU activation
#'
#' GELU approximation used by OpenAI CLIP: x * sigmoid(1.702 * x)
#' @param x Input tensor
#' @keywords internal
quick_gelu <- function(x) {
  x * torch::torch_sigmoid(1.702 * x)
}

#' CLIP MLP Block
#'
#' Feed-forward network with configurable activation
#' @param in_dim Input dimension
#' @param hidden_dim Hidden dimension
#' @param gelu_type GELU variant: "tanh" (tanh approximation), "quick" (QuickGELU), "exact" (standard GELU)
#' @keywords internal
CLIPMLP <- torch::nn_module(
  "CLIPMLP",

  initialize = function(in_dim, hidden_dim, gelu_type = "tanh") {
    self$fc1 <- torch::nn_linear(in_dim, hidden_dim)
    self$fc2 <- torch::nn_linear(hidden_dim, in_dim)
    self$gelu_type <- gelu_type
  },

  forward = function(x) {
    x <- self$fc1(x)
    if (self$gelu_type == "quick") {
      x <- quick_gelu(x)
    } else if (self$gelu_type == "tanh") {
      x <- torch::nnf_gelu(x, approximate = "tanh")
    } else {
      x <- torch::nnf_gelu(x)
    }
    x <- self$fc2(x)
    x
  }
)

#' CLIP Attention Block
#'
#' Multi-head self-attention with separate Q/K/V projections (HuggingFace style)
#' @param embed_dim Embedding dimension
#' @param num_heads Number of attention heads
#' @keywords internal
CLIPAttention <- torch::nn_module(
  "CLIPAttention",

  initialize = function(embed_dim, num_heads) {
    self$embed_dim <- embed_dim
    self$num_heads <- num_heads
    self$head_dim <- embed_dim %/% num_heads
    self$scale <- self$head_dim^(-0.5)

    # Separate Q/K/V projections (HuggingFace CLIPTextModel style)
    self$q_proj <- torch::nn_linear(embed_dim, embed_dim)
    self$k_proj <- torch::nn_linear(embed_dim, embed_dim)
    self$v_proj <- torch::nn_linear(embed_dim, embed_dim)
    self$out_proj <- torch::nn_linear(embed_dim, embed_dim)
  },

  forward = function(x, causal_mask = TRUE) {
    batch_size <- x$shape[1]
    seq_len <- x$shape[2]

    # Separate projections
    q <- self$q_proj(x)
    k <- self$k_proj(x)
    v <- self$v_proj(x)

    # Reshape for multi-head attention
    q <- q$view(c(batch_size, seq_len, self$num_heads, self$head_dim))$transpose(2, 3)
    k <- k$view(c(batch_size, seq_len, self$num_heads, self$head_dim))$transpose(2, 3)
    v <- v$view(c(batch_size, seq_len, self$num_heads, self$head_dim))$transpose(2, 3)

    # Scaled dot-product attention
    attn_weights <- torch::torch_matmul(q, k$transpose(3, 4)) * self$scale

    # Apply causal mask
    if (causal_mask) {
      mask <- torch::torch_ones(seq_len, seq_len, device = x$device)$triu(diagonal = 1)$bool()
      attn_weights <- attn_weights$masked_fill(mask, -Inf)
    }

    attn_weights <- torch::nnf_softmax(attn_weights, dim = -1)
    attn_output <- torch::torch_matmul(attn_weights, v)

    # Reshape back
    attn_output <- attn_output$transpose(2, 3)$contiguous()$view(c(batch_size, seq_len, self$embed_dim))
    attn_output <- self$out_proj(attn_output)

    attn_output
  }
)

#' CLIP Transformer Block
#'
#' Pre-norm transformer block with attention and MLP (HuggingFace style)
#' @param embed_dim Embedding dimension
#' @param num_heads Number of attention heads
#' @param mlp_dim MLP hidden dimension
#' @param gelu_type GELU variant: "tanh", "quick", or "exact"
#' @keywords internal
CLIPTransformerBlock <- torch::nn_module(
  "CLIPTransformerBlock",

  initialize = function(embed_dim, num_heads, mlp_dim, gelu_type = "tanh") {
    self$attention <- CLIPAttention(embed_dim, num_heads)
    self$layernorm_1 <- torch::nn_layer_norm(embed_dim)
    self$mlp <- CLIPMLP(embed_dim, mlp_dim, gelu_type)
    self$layernorm_2 <- torch::nn_layer_norm(embed_dim)
  },

  forward = function(x) {
    # Pre-norm attention: norm -> attn -> residual
    residual <- x
    x <- self$layernorm_1(x)
    x <- residual + self$attention(x)

    # Pre-norm MLP: norm -> mlp -> residual
    residual <- x
    x <- self$layernorm_2(x)
    x <- residual + self$mlp(x)

    x
  }
)

#' Native CLIP Text Encoder
#'
#' Native R torch implementation of CLIP text encoder.
#' Replaces TorchScript for better GPU compatibility.
#'
#' @param vocab_size Vocabulary size (default 49408)
#' @param context_length Maximum sequence length (default 77)
#' @param embed_dim Embedding dimension
#' @param num_layers Number of transformer layers
#' @param num_heads Number of attention heads
#' @param mlp_dim MLP hidden dimension
#' @param apply_final_ln Whether to apply final layer norm (default TRUE).
#'   Set to FALSE to match TorchScript exports that don't include final LN.
#'
#' @return An nn_module representing the text encoder
#' @export
text_encoder_native <- torch::nn_module(
  "TextEncoderNative",

  initialize = function(vocab_size = 49408, context_length = 77,
                        embed_dim = 768, num_layers = 12,
                        num_heads = 12, mlp_dim = 3072,
                        apply_final_ln = TRUE) {
    self$context_length <- context_length
    self$embed_dim <- embed_dim
    self$num_layers <- num_layers
    self$apply_final_ln <- apply_final_ln

    # Embeddings
    self$token_embedding <- torch::nn_embedding(vocab_size, embed_dim)
    self$position_embedding <- torch::nn_parameter(
      torch::torch_zeros(context_length, embed_dim)
    )

    # Transformer blocks - tanh GELU approximation matches TorchScript export best
    self$transformer_blocks <- torch::nn_module_list()
    for (i in seq_len(num_layers)) {
      self$transformer_blocks$append(
        CLIPTransformerBlock(embed_dim, num_heads, mlp_dim, gelu_type = "tanh")
      )
    }

    # Final layer norm
    self$final_layer_norm <- torch::nn_layer_norm(embed_dim)
  },

  forward = function(input_ids) {
    # Move input to model's device
    input_ids <- input_ids$to(device = self$token_embedding$weight$device)

    batch_size <- input_ids$shape[1]
    seq_length <- input_ids$shape[2]

    # Token + position embeddings
    # Add 1 because R torch is 1-indexed but tokens are 0-indexed (Python convention)
    token_embeds <- self$token_embedding(input_ids + 1L)
    pos_embeds <- self$position_embedding[1:seq_length, ]$unsqueeze(1)$expand(c(batch_size, -1, -1))
    hidden_states <- token_embeds + pos_embeds

    # Transformer layers
    for (i in seq_len(self$num_layers)) {
      hidden_states <- self$transformer_blocks[[i]](hidden_states)
    }

    # Conditionally apply final layer norm to match TorchScript behavior
    if (self$apply_final_ln) {
      hidden_states <- self$final_layer_norm(hidden_states)
    }

    hidden_states
  }
)

#' Detect text encoder architecture from TorchScript file
#'
#' @param torchscript_path Path to TorchScript encoder .pt file
#' @return List with vocab_size, context_length, embed_dim, num_layers, num_heads, mlp_dim
#' @keywords internal
detect_text_encoder_architecture <- function(torchscript_path) {
  ts_encoder <- torch::jit_load(torchscript_path)
  ts_params <- ts_encoder$parameters
  param_names <- names(ts_params)

  # Detect prefix style (SD21 vs SDXL)
  if (any(grepl("^text_encoder\\.", param_names))) {
    prefix <- "text_encoder.text_model."
  } else if (any(grepl("^enc\\.", param_names))) {
    prefix <- "enc.text_model."
  } else {
    stop("Unknown TorchScript parameter prefix")
  }

  # Get dimensions from embeddings
  tok_emb <- ts_params[[paste0(prefix, "embeddings.token_embedding.weight")]]
  pos_emb <- ts_params[[paste0(prefix, "embeddings.position_embedding.weight")]]
  fc1 <- ts_params[[paste0(prefix, "encoder.layers.0.mlp.fc1.weight")]]

  vocab_size <- as.integer(tok_emb$shape[1])
  embed_dim <- as.integer(tok_emb$shape[2])
  context_length <- as.integer(pos_emb$shape[1])
  mlp_dim <- as.integer(fc1$shape[1])

  # Count layers
  layer_params <- grep("encoder\\.layers\\.[0-9]+\\.", param_names, value = TRUE)
  layer_nums <- unique(as.integer(gsub(".*encoder\\.layers\\.([0-9]+)\\..*", "\\1", layer_params)))
  num_layers <- length(layer_nums)

  # Infer heads from embed_dim (typical head_dim is 64)
  num_heads <- embed_dim %/% 64L

  # Detect if final layer norm is applied in TorchScript output
  # by checking output range with test input
  tokens <- torch::torch_tensor(matrix(c(49406, 320, 4380, 49407, rep(49407, 73)), nrow = 1),
                                 dtype = torch::torch_long())
  torch::with_no_grad({
    test_out <- ts_encoder(tokens)
  })
  # Handle both single tensor (text_encoder) and list output (text_encoder2)
  if (is.list(test_out)) {
    hidden_states <- test_out[[1]]
  } else {
    hidden_states <- test_out
  }
  # If output range is large (>100), final LN is NOT applied
  max_abs <- max(abs(as.numeric(hidden_states$min())), abs(as.numeric(hidden_states$max())))
  apply_final_ln <- max_abs < 100

  list(
    vocab_size = vocab_size,
    context_length = context_length,
    embed_dim = embed_dim,
    num_layers = num_layers,
    num_heads = num_heads,
    mlp_dim = mlp_dim,
    prefix = prefix,
    apply_final_ln = apply_final_ln
  )
}

#' Load weights from TorchScript text encoder into native encoder
#'
#' @param native_encoder Native text encoder module
#' @param torchscript_path Path to TorchScript encoder .pt file
#' @param verbose Print loading progress
#'
#' @return The native encoder with loaded weights (invisibly)
#' @export
load_text_encoder_weights <- function(native_encoder, torchscript_path, verbose = TRUE) {
  ts_encoder <- torch::jit_load(torchscript_path)
  ts_params <- ts_encoder$parameters
  param_names <- names(ts_params)

  # Build mapping from TorchScript names to native names
  # TorchScript format: enc.text_model.encoder.layers.0.self_attn.q_proj.weight
  #                or: text_encoder.text_model.encoder.layers.0.self_attn.q_proj.weight
  # Native format: transformer_blocks.1.attention.q_proj.weight
  remap_key <- function(key) {
    # Strip text_encoder. prefix (SD21 style)
    key <- sub("^text_encoder\\.", "", key)

    # Strip enc. prefix (SDXL style)
    key <- sub("^enc\\.", "", key)

    # Strip text_model. prefix
    key <- sub("^text_model\\.", "", key)

    # Embeddings
    key <- sub("^embeddings\\.token_embedding\\.", "token_embedding.", key)
    # position_embedding.weight -> position_embedding (it's an nn_parameter, not nn_embedding)
    key <- sub("^embeddings\\.position_embedding\\.weight$", "position_embedding", key)

    # Encoder layers -> transformer_blocks (keep 0-indexed)
    key <- gsub("^encoder\\.layers\\.", "transformer_blocks.", key)

    # self_attn -> attention
    key <- gsub("\\.self_attn\\.", ".attention.", key)

    # Layer norms
    key <- gsub("\\.layer_norm1\\.", ".layernorm_1.", key)
    key <- gsub("\\.layer_norm2\\.", ".layernorm_2.", key)

    # Final layer norm
    key <- sub("^final_layer_norm\\.", "final_layer_norm.", key)

    # MLP (HuggingFace uses fc1/fc2 already)
    # No change needed

    key
  }

  loaded <- 0
  skipped <- 0
  unmapped <- character(0)

  torch::with_no_grad({
    for (ts_name in names(ts_params)) {
      native_name <- remap_key(ts_name)

      if (native_name %in% names(native_encoder$parameters)) {
        ts_tensor <- ts_params[[ts_name]]
        native_tensor <- native_encoder$parameters[[native_name]]

        if (all(ts_tensor$shape == native_tensor$shape)) {
          native_tensor$copy_(ts_tensor)
          loaded <- loaded + 1
        } else if (verbose) {
          message("Shape mismatch: ", native_name,
                  " (", paste(as.integer(ts_tensor$shape), collapse="x"), " vs ",
                  paste(as.integer(native_tensor$shape), collapse="x"), ")")
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
      message("Unmapped: ", length(unmapped), " parameters (showing first 5)")
      for (u in head(unmapped, 5)) message("  ", u)
    }
    message("Loaded ", loaded, "/", loaded + skipped, " parameters")
  }

  invisible(native_encoder)
}

#' Native CLIP Text Encoder 2 (OpenCLIP ViT-bigG for SDXL)
#'
#' Native R torch implementation of OpenCLIP text encoder used in SDXL.
#' Returns both hidden states and pooled output.
#'
#' @param vocab_size Vocabulary size (default 49408)
#' @param context_length Maximum sequence length (default 77)
#' @param embed_dim Embedding dimension (default 1280)
#' @param num_layers Number of transformer layers (default 32)
#' @param num_heads Number of attention heads (default 20)
#' @param mlp_dim MLP hidden dimension (default 5120)
#'
#' @return An nn_module representing the text encoder
#' @export
text_encoder2_native <- torch::nn_module(
  "TextEncoder2Native",

  initialize = function(vocab_size = 49408, context_length = 77,
                        embed_dim = 1280, num_layers = 32,
                        num_heads = 20, mlp_dim = 5120) {
    self$context_length <- context_length
    self$embed_dim <- embed_dim
    self$num_layers <- num_layers

    # Embeddings
    self$token_embedding <- torch::nn_embedding(vocab_size, embed_dim)
    self$position_embedding <- torch::nn_parameter(
      torch::torch_zeros(context_length, embed_dim)
    )

    # Transformer blocks with standard GELU (OpenCLIP style)
    self$transformer_blocks <- torch::nn_module_list()
    for (i in seq_len(num_layers)) {
      self$transformer_blocks$append(
        CLIPTransformerBlock(embed_dim, num_heads, mlp_dim, gelu_type = "exact")
      )
    }

    # Final layer norm
    self$final_layer_norm <- torch::nn_layer_norm(embed_dim)

    # Text projection for pooled output
    self$text_projection <- torch::nn_linear(embed_dim, embed_dim, bias = FALSE)
  },

  forward = function(input_ids) {
    # Move input to model's device
    input_ids <- input_ids$to(device = self$token_embedding$weight$device)

    batch_size <- input_ids$shape[1]
    seq_length <- input_ids$shape[2]

    # Token + position embeddings
    # Add 1 because R torch is 1-indexed but tokens are 0-indexed (Python convention)
    token_embeds <- self$token_embedding(input_ids + 1L)
    pos_embeds <- self$position_embedding[1:seq_length, ]$unsqueeze(1)$expand(c(batch_size, -1, -1))
    hidden_states <- token_embeds + pos_embeds

    # Transformer layers
    for (i in seq_len(self$num_layers)) {
      hidden_states <- self$transformer_blocks[[i]](hidden_states)
    }

    # Note: TorchScript does NOT apply final_layer_norm to hidden_states output
    # Only apply it for pooled output computation
    hidden_states_normalized <- self$final_layer_norm(hidden_states)

    # Pooled output: take the EOS token embedding (last non-padded token)
    # For simplicity, use the position of the max token ID (EOS = 49407)
    # In practice, SDXL tokenizer puts EOS at the position after the last real token
    eos_indices <- torch::torch_argmax(input_ids, dim = 2L, keepdim = TRUE)
    pooled_output <- hidden_states_normalized$gather(
      dim = 2L,
      index = eos_indices$unsqueeze(-1L)$expand(c(-1L, -1L, self$embed_dim))
    )$squeeze(2L)

    # Apply text projection
    pooled_output <- self$text_projection(pooled_output)

    # Return hidden_states WITHOUT final LN (matches TorchScript), pooled WITH LN + projection
    list(hidden_states, pooled_output)
  }
)

#' Load weights from TorchScript text encoder 2 into native encoder
#'
#' @param native_encoder Native text encoder 2 module
#' @param torchscript_path Path to TorchScript encoder .pt file
#' @param verbose Print loading progress
#'
#' @return The native encoder with loaded weights (invisibly)
#' @export
load_text_encoder2_weights <- function(native_encoder, torchscript_path, verbose = TRUE) {
  ts_encoder <- torch::jit_load(torchscript_path)
  ts_params <- ts_encoder$parameters

  # Build mapping from TorchScript names to native names
  # TorchScript format: enc.text_model.encoder.layers.0.self_attn.q_proj.weight
  # Native format: transformer_blocks.1.attention.q_proj.weight
  remap_key <- function(key) {
    # Strip enc. prefix
    key <- sub("^enc\\.", "", key)

    # text_projection is at enc level, not text_model level
    if (grepl("^text_projection", key)) {
      return(key)
    }

    # Strip text_model. prefix
    key <- sub("^text_model\\.", "", key)

    # Embeddings
    key <- sub("^embeddings\\.token_embedding\\.", "token_embedding.", key)
    # position_embedding.weight -> position_embedding (it's an nn_parameter, not nn_embedding)
    key <- sub("^embeddings\\.position_embedding\\.weight$", "position_embedding", key)

    # Encoder layers -> transformer_blocks (keep 0-indexed)
    key <- gsub("^encoder\\.layers\\.", "transformer_blocks.", key)

    # self_attn -> attention
    key <- gsub("\\.self_attn\\.", ".attention.", key)

    # Layer norms
    key <- gsub("\\.layer_norm1\\.", ".layernorm_1.", key)
    key <- gsub("\\.layer_norm2\\.", ".layernorm_2.", key)

    # Final layer norm
    key <- sub("^final_layer_norm\\.", "final_layer_norm.", key)

    # MLP (HuggingFace uses fc1/fc2 already)
    # No change needed

    key
  }

  loaded <- 0
  skipped <- 0
  unmapped <- character(0)

  torch::with_no_grad({
    for (ts_name in names(ts_params)) {
      native_name <- remap_key(ts_name)

      if (native_name %in% names(native_encoder$parameters)) {
        ts_tensor <- ts_params[[ts_name]]
        native_tensor <- native_encoder$parameters[[native_name]]

        if (all(ts_tensor$shape == native_tensor$shape)) {
          native_tensor$copy_(ts_tensor)
          loaded <- loaded + 1
        } else if (verbose) {
          message("Shape mismatch: ", native_name,
                  " (", paste(as.integer(ts_tensor$shape), collapse="x"), " vs ",
                  paste(as.integer(native_tensor$shape), collapse="x"), ")")
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
      message("Unmapped: ", length(unmapped), " parameters (showing first 5)")
      for (u in head(unmapped, 5)) message("  ", u)
    }
    message("Loaded ", loaded, "/", loaded + skipped, " parameters")
  }

  invisible(native_encoder)
}
