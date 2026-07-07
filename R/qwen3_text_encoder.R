#' Qwen3 Text Encoder
#'
#' Fresh R port of the Qwen3 decoder stack from HuggingFace transformers
#' (Apache-2.0, src/transformers/models/qwen3/), used by FLUX.2 klein as
#' its text encoder (Qwen3-4B: 36 layers, hidden 2560, 32 query / 8 KV
#' heads, head_dim 128, SwiGLU 9728, RoPE theta 1e6). The pipeline
#' consumes mid-stack hidden states (layers 9, 18, 27 for klein-4B)
#' concatenated per token, so the forward runs only as deep as the last
#' requested layer; the LM head is never needed (embeddings are tied).
#' Causal attention with the tokenizer's padding mask, matching the
#' reference exactly.
#'
#' @name qwen3_text_encoder
NULL

# Llama-convention rotary tables: cos/sin [S, head_dim/2] at the given
# theta, applied with the split-half kernel (x1*cos - x2*sin | x2*cos +
# x1*sin), which matches rotate_half exactly.
.qwen3_rope_tables <- function(seq_len, head_dim, theta, device) {
    f32 <- torch::torch_float32()
    inv_freq <- 1.0 / torch::torch_pow(
                                       theta,
                                       torch::torch_arange(start = 0, end = head_dim - 2, step = 2,
            dtype = f32, device = device) / head_dim
    )
    pos <- torch::torch_arange(start = 0, end = seq_len - 1, dtype = f32,
                               device = device)
    freqs <- pos$unsqueeze(2L) * inv_freq$unsqueeze(1L) # [S, D/2]
    # [1, 1, S, D/2] for broadcasting against [B, H, S, D/2]
    list(
         freqs$cos()$unsqueeze(1L)$unsqueeze(1L),
         freqs$sin()$unsqueeze(1L)$unsqueeze(1L)
    )
}

.qwen3_attention <- torch::nn_module(
                                     "qwen3_attention",
                                     initialize = function(hidden_size, num_heads, num_kv_heads, head_dim,
        eps = 1e-6) {
    self$num_heads <- num_heads
    self$num_kv_heads <- num_kv_heads
    self$head_dim <- head_dim
    self$q_proj <- torch::nn_linear(hidden_size, num_heads * head_dim,
                                    bias = FALSE)
    self$k_proj <- torch::nn_linear(hidden_size, num_kv_heads * head_dim,
                                    bias = FALSE)
    self$v_proj <- torch::nn_linear(hidden_size, num_kv_heads * head_dim,
                                    bias = FALSE)
    self$o_proj <- torch::nn_linear(num_heads * head_dim, hidden_size,
                                    bias = FALSE)
    self$q_norm <- ltx23_rms_norm(head_dim, eps = eps)
    self$k_norm <- ltx23_rms_norm(head_dim, eps = eps)
},
                                     forward = function(x, rope, mask = NULL) {
    shape <- x$shape
    b <- shape[1]
    s <- shape[2]

    q <- self$q_proj(x)$view(c(b, s, self$num_heads, self$head_dim))
    k <- self$k_proj(x)$view(c(b, s, self$num_kv_heads, self$head_dim))
    v <- self$v_proj(x)$view(c(b, s, self$num_kv_heads, self$head_dim))

    q <- self$q_norm(q)$transpose(2L, 3L) # [B, H, S, D]
    k <- self$k_norm(k)$transpose(2L, 3L)
    v <- v$transpose(2L, 3L)

    q <- ltx23_apply_split_rotary_emb(q, rope)
    k <- ltx23_apply_split_rotary_emb(k, rope)

    # GQA: repeat KV heads to match the query heads
    groups <- self$num_heads %/% self$num_kv_heads
    if (groups > 1L) {
        k <- k$repeat_interleave(groups, dim = 2L)
        v <- v$repeat_interleave(groups, dim = 2L)
    }

    out <- .ltx23_sdpa(q, k, v, attention_mask = mask)
    out <- out$transpose(2L, 3L)$reshape(c(b, s, -1L))
    self$o_proj(out)
}
)

.qwen3_mlp <- torch::nn_module(
                               "qwen3_mlp",
                               initialize = function(hidden_size, intermediate_size) {
    self$gate_proj <- torch::nn_linear(hidden_size, intermediate_size,
                                       bias = FALSE)
    self$up_proj <- torch::nn_linear(hidden_size, intermediate_size,
                                     bias = FALSE)
    self$down_proj <- torch::nn_linear(intermediate_size, hidden_size,
                                       bias = FALSE)
},
                               forward = function(x) {
    self$down_proj(torch::nnf_silu(self$gate_proj(x)) * self$up_proj(x))
}
)

.qwen3_layer <- torch::nn_module(
                                 "qwen3_layer",
                                 initialize = function(hidden_size, num_heads, num_kv_heads, head_dim,
        intermediate_size, eps = 1e-6) {
    self$self_attn <- .qwen3_attention(hidden_size, num_heads, num_kv_heads,
                                       head_dim, eps = eps)
    self$mlp <- .qwen3_mlp(hidden_size, intermediate_size)
    self$input_layernorm <- ltx23_rms_norm(hidden_size, eps = eps)
    self$post_attention_layernorm <- ltx23_rms_norm(hidden_size, eps = eps)
},
                                 forward = function(x, rope, mask = NULL) {
    x <- x + self$self_attn(self$input_layernorm(x), rope, mask)
    x + self$mlp(self$post_attention_layernorm(x))
}
)

#' Qwen3 encoder stack
#'
#' Defaults are the Qwen3-4B configuration used by FLUX.2 klein. The
#' module tree mirrors the checkpoint keys (\code{model.embed_tokens},
#' \code{model.layers.*}, \code{model.norm}); the tied LM head carries
#' no weights of its own and is not implemented.
#'
#' @param vocab_size,hidden_size,intermediate_size,num_hidden_layers
#'   Integers.
#' @param num_attention_heads,num_key_value_heads,head_dim Integers.
#' @param rope_theta Numeric.
#' @param rms_norm_eps Numeric.
#'
#' @return Module whose forward(input_ids, attention_mask = NULL,
#'   out_layers) returns a list of hidden-state tensors [B, S, hidden],
#'   one per requested layer (a value of k means the state after k
#'   layers, matching HF \code{output.hidden_states[k]}). Runs only to
#'   \code{max(out_layers)}. \code{input_ids} are 1-based.
#'
#' @export
qwen3_encoder <- torch::nn_module(
                                  "qwen3_encoder",
                                  initialize = function(vocab_size = 151936L, hidden_size = 2560L,
        intermediate_size = 9728L,
        num_hidden_layers = 36L,
        num_attention_heads = 32L,
        num_key_value_heads = 8L, head_dim = 128L,
        rope_theta = 1e6, rms_norm_eps = 1e-6) {
    self$head_dim <- head_dim
    self$rope_theta <- rope_theta

    inner <- torch::nn_module(
                              "qwen3_model",
                              initialize = function() {
        self$embed_tokens <- torch::nn_embedding(vocab_size, hidden_size)
        self$layers <- torch::nn_module_list(
            lapply(seq_len(num_hidden_layers), function(i) {
            .qwen3_layer(hidden_size, num_attention_heads,
                         num_key_value_heads, head_dim, intermediate_size,
                         eps = rms_norm_eps)
        })
        )
        self$norm <- ltx23_rms_norm(hidden_size, eps = rms_norm_eps)
    }
    )
    self$model <- inner()
},
                                  forward = function(input_ids, attention_mask = NULL,
        out_layers = c(9L, 18L, 27L)) {
    x <- self$model$embed_tokens(input_ids)
    b <- input_ids$shape[1]
    s <- input_ids$shape[2]
    device <- x$device
    f32 <- torch::torch_float32()

    rope <- .qwen3_rope_tables(s, self$head_dim, self$rope_theta, device)

    # Additive causal (+ padding) mask [B, 1, S, S] in float32
    neg <- -3.4e38
    causal <- torch::torch_full(c(s, s), neg, dtype = f32,
                                device = device)$triu(diagonal = 1L)
    mask <- causal$unsqueeze(1L)$unsqueeze(1L)$expand(c(b, 1L, s, s))
    if (!is.null(attention_mask)) {
        pad <- (1 - attention_mask$to(dtype = f32, device = device))$mul(neg)
        mask <- mask + pad$unsqueeze(2L)$unsqueeze(2L)
    }

    out_layers <- sort(as.integer(out_layers))
    states <- vector("list", length(out_layers))
    for (i in seq_len(max(out_layers))) {
        x <- self$model$layers[[i]](x, rope, mask)
        hit <- which(out_layers == i)
        if (length(hit)) {
            states[[hit]] <- x
        }
    }
    states
}
)

#' Load a Qwen3 encoder from a transformers directory
#'
#' Streams the (possibly sharded) safetensors weights into
#' \code{\link{qwen3_encoder}}. The LM head is tied to the embeddings
#' and skipped.
#'
#' @param model_path Directory with \code{config.json} and
#'   \code{model*.safetensors} (FLUX.2-klein's \code{text_encoder}).
#' @param device Character. Target device.
#' @param dtype Character. "bfloat16" (GPU) or "float32" (CPU).
#' @param verbose Logical.
#' @param ... Overrides for \code{\link{qwen3_encoder}} arguments.
#'
#' @return The loaded \code{qwen3_encoder} in eval mode.
#'
#' @export
load_qwen3_text_encoder <- function(model_path, device = "cpu",
                                    dtype = "float32", verbose = TRUE, ...) {
    model_path <- path.expand(model_path)
    config <- NULL
    config_path <- file.path(model_path, "config.json")
    if (file.exists(config_path)) {
        config <- jsonlite::fromJSON(config_path, simplifyVector = TRUE)
    }

    args <- list(vocab_size = config$vocab_size,
                 hidden_size = config$hidden_size,
                 intermediate_size = config$intermediate_size,
                 num_hidden_layers = config$num_hidden_layers,
                 num_attention_heads = config$num_attention_heads,
                 num_key_value_heads = config$num_key_value_heads,
                 head_dim = config$head_dim)
    args <- Filter(function(x) !is.null(x) && length(x) > 0L, args)
    args <- lapply(args, as.integer)
    if (!is.null(config$rope_theta)) {
        args$rope_theta <- as.numeric(config$rope_theta)
    }
    if (!is.null(config$rms_norm_eps)) {
        args$rms_norm_eps <- as.numeric(config$rms_norm_eps)
    }
    args <- utils::modifyList(args, list(...))
    model <- do.call(qwen3_encoder, args)
    model$to(dtype = .flux_dtype(dtype))

    opened <- .flux_open_sharded_dir(model_path, "model")
    ckpt <- structure(
                      list(handle = opened$handle, keys = opened$keys,
                           version = NULL, config = config, path = model_path),
                      class = "ltx23_checkpoint"
    )
    map_key <- function(key) {
        if (identical(key, "lm_head.weight")) {
            return(NA_character_) # tied to embed_tokens
        }
        key
    }
    res <- ltx23_load_group(ckpt, ckpt$keys, model, map_key = map_key,
                            verbose = verbose)
    if (length(res$unmapped) || length(res$unfilled)) {
        stop("Qwen3 encoder load: ", length(res$unmapped),
             " unmapped keys, ", length(res$unfilled), " unfilled params")
    }

    model$to(device = device)
    model$eval()
    model
}

#' Encode prompts with the Qwen3 encoder for FLUX.2
#'
#' Tokenizes with the chat template, runs the encoder with the padding
#' mask, and concatenates the requested mid-stack hidden states per
#' token, matching Flux2KleinPipeline._get_qwen3_prompt_embeds.
#'
#' @param prompts Character vector.
#' @param model A \code{\link{qwen3_encoder}}.
#' @param tokenizer A \code{\link{qwen_bpe_tokenizer}}.
#' @param max_sequence_length Integer. Fixed token length (klein: 512).
#' @param out_layers Integer vector. Hidden-state layers (klein-4B:
#'   9, 18, 27).
#' @param device Device for the input ids (defaults to the model's).
#'
#' @return Tensor [length(prompts), max_sequence_length,
#'   3 * hidden_size].
#'
#' @export
encode_with_qwen3 <- function(prompts, model, tokenizer,
                              max_sequence_length = 512L,
                              out_layers = c(9L, 18L, 27L), device = NULL) {
    enc <- encode_qwen(tokenizer, prompts, max_length = max_sequence_length,
                       chat_template = TRUE)
    device <- device %||% model$model$embed_tokens$weight$device
    long <- torch::torch_long()
    ids <- torch::torch_tensor(enc$input_ids + 1L, dtype = long,
                               device = device)
    mask <- torch::torch_tensor(enc$attention_mask, dtype = long,
                                device = device)

    states <- torch::with_no_grad(model(ids, attention_mask = mask,
                                        out_layers = out_layers))
    # stack(dim=1) + permute + reshape == per-token concatenation
    torch::torch_cat(states, dim = -1L)
}
