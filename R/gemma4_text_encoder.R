# Gemma 4 Text Encoder for LTX-2.5
#
# Native R torch port of the text stack of HuggingFace transformers'
# Gemma4UnifiedTextModel (Apache-2.0, models/gemma4_unified). LTX-2.5
# ships a fine-tuned 12B (gemma4-12b-ltx-v1) whose trunk matches the
# Gemma 3 12B of LTX-2.3 (48 layers, hidden 3840, 5 sliding : 1 full),
# with these differences:
# - RMSNorm scales by w (Gemma 3: 1 + w); values get a scale-free RMSNorm
# - attention scaling is 1.0 (the q/k norms carry the scale)
# - full-attention layers: head_dim 512, one KV head, keys reused as
#   values (attention_k_eq_v), "proportional" RoPE rotating a quarter of
#   the dims at theta 1e6
# - every layer multiplies its output by a stored layer_scalar
# Vision/audio towers and the KV-sharing / per-layer-input / MoE options
# are off in this checkpoint and not ported.

#' Gemma 4 RMS normalization
#'
#' @param dim Integer. Normalized dimension.
#' @param eps Numeric. Epsilon.
#' @param with_scale Logical. Carry a learned scale (FALSE for the value
#'   norm).
#'
#' @return Module whose forward(x) normalizes over the last dimension in
#'   float32 and multiplies by the scale, returning \code{x}'s dtype.
#'
#' @keywords internal
gemma4_rms_norm <- torch::nn_module(
                                    "Gemma4RMSNorm",
                                    initialize = function(dim, eps = 1e-6, with_scale = TRUE) {
    self$eps <- eps
    self$with_scale <- with_scale
    if (with_scale) {
        self$weight <- torch::nn_parameter(torch::torch_ones(dim))
    }
},
                                    forward = function(x) {
    xf <- x$to(dtype = torch::torch_float32())
    out <- xf * xf$pow(2)$mean(dim = -1L, keepdim = TRUE)$add(self$eps)$pow(-0.5)
    if (self$with_scale) {
        out <- out * self$weight$to(dtype = torch::torch_float32())
    }
    out$to(dtype = x$dtype)
}
)

# Inverse RoPE frequencies for one layer type. "proportional" rotates
# the first floor(partial * head_dim / 2) frequency pairs, computed over
# the full head_dim, and leaves the rest unrotated (zero frequency).
.gemma4_inv_freq <- function(head_dim, params) {
    base <- params$rope_theta
    if (identical(params$rope_type %||% "default", "proportional")) {
        angles <- as.integer(floor((params$partial_rotary_factor %||% 1) *
                                   head_dim / 2))
        rotated <- 1 / base^(seq(0, 2 * angles - 1, by = 2) / head_dim)
        c(rotated, rep(0, head_dim %/% 2L - angles))
    } else {
        1 / base^(seq(0, head_dim - 1, by = 2) / head_dim)
    }
}

# cos/sin [B, S, head_dim] for integer positions [B, S]
.gemma4_rope <- function(inv_freq, position_ids, dtype) {
    inv <- torch::torch_tensor(inv_freq, dtype = torch::torch_float32(),
                               device = position_ids$device)
    freqs <- position_ids$to(dtype = torch::torch_float32())$unsqueeze(-1L) * inv
    emb <- torch::torch_cat(list(freqs, freqs), dim = -1L)
    list(emb$cos()$to(dtype = dtype), emb$sin()$to(dtype = dtype))
}

#' Gemma 4 text attention
#'
#' @param config Model configuration (see \code{\link{load_gemma4_text_encoder}}).
#' @param layer_idx Integer. 0-based layer index.
#'
#' @return Module whose forward(hidden_states, attention_mask, cos, sin)
#'   returns the attention output [B, S, hidden_size].
#'
#' @keywords internal
gemma4_attention <- torch::nn_module(
                                     "Gemma4Attention",
                                     initialize = function(config, layer_idx) {
    full <- identical(config$layer_types[[layer_idx + 1L]], "full_attention")
    self$is_sliding <- !full
    self$num_heads <- config$num_attention_heads
    self$head_dim <- if (full) config$global_head_dim else config$head_dim
    self$k_eq_v <- isTRUE(config$attention_k_eq_v) && full
    self$num_kv_heads <- if (self$k_eq_v) {
        config$num_global_key_value_heads %||% config$num_key_value_heads
    } else {
        config$num_key_value_heads
    }
    eps <- config$rms_norm_eps
    h <- config$hidden_size

    self$q_proj <- linear_noinit(h, self$num_heads * self$head_dim, bias = FALSE)
    self$k_proj <- linear_noinit(h, self$num_kv_heads * self$head_dim, bias = FALSE)
    if (!self$k_eq_v) {
        self$v_proj <- linear_noinit(h, self$num_kv_heads * self$head_dim,
                                     bias = FALSE)
    }
    self$o_proj <- linear_noinit(self$num_heads * self$head_dim, h, bias = FALSE)
    self$q_norm <- gemma4_rms_norm(self$head_dim, eps = eps)
    self$k_norm <- gemma4_rms_norm(self$head_dim, eps = eps)
    self$v_norm <- gemma4_rms_norm(self$head_dim, eps = eps, with_scale = FALSE)
},
                                     forward = function(hidden_states, attention_mask, cos, sin) {
    b <- hidden_states$shape[1]
    s <- hidden_states$shape[2]
    heads <- function(x, n) x$view(c(b, s, n, self$head_dim))
    cos <- cos$unsqueeze(3L)
    sin <- sin$unsqueeze(3L)
    rope <- function(x) x * cos + rotate_half(x) * sin

    q <- rope(self$q_norm(heads(self$q_proj(hidden_states), self$num_heads)))
    k_raw <- heads(self$k_proj(hidden_states), self$num_kv_heads)
    # k == v layers take the values from the raw key projection
    v_raw <- if (self$k_eq_v) {
        k_raw
    } else {
        heads(self$v_proj(hidden_states), self$num_kv_heads)
    }
    k <- rope(self$k_norm(k_raw))
    v <- self$v_norm(v_raw)

    q <- q$transpose(2L, 3L)
    k <- repeat_kv(k$transpose(2L, 3L), self$num_heads %/% self$num_kv_heads)
    v <- repeat_kv(v$transpose(2L, 3L), self$num_heads %/% self$num_kv_heads)
    out <- torch::torch_scaled_dot_product_attention(q, k, v,
                                                     attn_mask = attention_mask,
                                                     scale = 1.0)
    out <- out$transpose(2L, 3L)$reshape(c(b, s, self$num_heads * self$head_dim))
    self$o_proj(out)
}
)

#' Gemma 4 decoder layer
#'
#' @param config Model configuration.
#' @param layer_idx Integer. 0-based layer index.
#'
#' @return Module whose forward(hidden_states, attention_mask, cos, sin)
#'   returns the layer output, scaled by the stored \code{layer_scalar}.
#'
#' @keywords internal
gemma4_decoder_layer <- torch::nn_module(
    "Gemma4DecoderLayer",
    initialize = function(config, layer_idx) {
    eps <- config$rms_norm_eps
    h <- config$hidden_size
    self$self_attn <- gemma4_attention(config, layer_idx)
    self$mlp <- gemma3_mlp(config)
    self$input_layernorm <- gemma4_rms_norm(h, eps = eps)
    self$post_attention_layernorm <- gemma4_rms_norm(h, eps = eps)
    self$pre_feedforward_layernorm <- gemma4_rms_norm(h, eps = eps)
    self$post_feedforward_layernorm <- gemma4_rms_norm(h, eps = eps)
    self$layer_scalar <- torch::nn_buffer(torch::torch_ones(1L))
},
    forward = function(hidden_states, attention_mask, cos, sin) {
    residual <- hidden_states
    x <- self$input_layernorm(hidden_states)
    x <- self$self_attn(x, attention_mask, cos, sin)
    hidden_states <- residual + self$post_attention_layernorm(x)

    residual <- hidden_states
    x <- self$mlp(self$pre_feedforward_layernorm(hidden_states))
    hidden_states <- residual + self$post_feedforward_layernorm(x)
    hidden_states * self$layer_scalar$to(dtype = hidden_states$dtype)
}
)

#' Gemma 4 text model
#'
#' The text stack of Gemma 4 Unified, as the LTX-2.5 text encoder.
#'
#' @param config Model configuration list: the checkpoint's
#'   \code{text_config} fields (see \code{\link{load_gemma4_text_encoder}}).
#'
#' @return Module whose forward(input_ids, attention_mask,
#'   output_hidden_states) returns \code{list(last_hidden_state,
#'   hidden_states)}. \code{input_ids} are 0-based; \code{hidden_states}
#'   follows transformers: the state entering each layer, then the
#'   final-normed output (\code{num_hidden_layers + 1} entries).
#'
#' @export
gemma4_text_model <- torch::nn_module(
                                      "Gemma4TextModel",
                                      initialize = function(config) {
    self$config <- config
    self$embed_tokens <- embedding_noinit(config$vocab_size, config$hidden_size)
    self$layers <- torch::nn_module_list(lapply(seq_len(config$num_hidden_layers) - 1L,
        function(i) gemma4_decoder_layer(config, i)))
    self$norm <- gemma4_rms_norm(config$hidden_size, eps = config$rms_norm_eps)
    rp <- config$rope_parameters
    self$inv_freq <- list(
        sliding_attention = .gemma4_inv_freq(config$head_dim, rp$sliding_attention),
        full_attention = .gemma4_inv_freq(config$global_head_dim, rp$full_attention)
    )
},
                                      forward = function(input_ids, attention_mask = NULL,
                                                         output_hidden_states = FALSE) {
    b <- input_ids$shape[1]
    s <- input_ids$shape[2]
    device <- input_ids$device
    w <- self$embed_tokens$weight
    # The embedding scale is rounded to the weight dtype, as in transformers
    # (sqrt(3840) = 61.97 becomes 62 in bfloat16)
    scale <- torch::torch_tensor(sqrt(self$config$hidden_size), dtype = w$dtype,
                                 device = w$device)
    hidden_states <- self$embed_tokens(input_ids + 1L) * scale
    dtype <- hidden_states$dtype

    masks <- .gemma4_masks(attention_mask, b, s, self$config$sliding_window,
                           dtype, device)
    positions <- torch::torch_arange(0, s - 1L, device = device)$unsqueeze(1L)$expand(c(b, -1L))
    rope <- lapply(self$inv_freq, .gemma4_rope, position_ids = positions,
                   dtype = dtype)

    all_hidden <- NULL
    # Layer streaming (load with stream = TRUE): each layer's weights
    # visit the device only for its own forward
    stream <- self$layer_stream
    for (i in seq_along(self$layers)) {
        if (output_hidden_states) {
            all_hidden <- c(all_hidden, list(hidden_states))
        }
        type <- self$config$layer_types[[i]]
        if (!is.null(stream)) {
            .staged_onload(stream[[i]], device)
        }
        hidden_states <- self$layers[[i]](hidden_states, masks[[type]],
                                          rope[[type]][[1]], rope[[type]][[2]])
        if (!is.null(stream)) {
            .staged_offload(stream[[i]])
            gc(verbose = FALSE, full = FALSE)
        }
    }
    hidden_states <- self$norm(hidden_states)
    if (output_hidden_states) {
        all_hidden <- c(all_hidden, list(hidden_states))
    }
    list(last_hidden_state = hidden_states, hidden_states = all_hidden)
}
)

# Additive [B, 1, S, S] masks per layer type: causal, keys limited to
# attention_mask, and for sliding layers q - k < window. Masked entries
# take the dtype's finite minimum, so a fully masked row (a left-pad
# query) softmaxes to a uniform finite row instead of NaN.
.gemma4_masks <- function(attention_mask, b, s, window, dtype, device) {
    q <- torch::torch_arange(0, s - 1L, device = device)$view(c(s, 1L))
    k <- torch::torch_arange(0, s - 1L, device = device)$view(c(1L, s))
    causal <- k$le(q)
    sliding <- causal$logical_and((q - k)$lt(window))
    keep <- if (is.null(attention_mask)) {
        torch::torch_ones(c(b, 1L, 1L, s), dtype = torch::torch_bool(),
                          device = device)
    } else {
        attention_mask$to(device = device)$gt(0)$view(c(b, 1L, 1L, s))
    }
    neg <- .gemma4_dtype_min(dtype)
    additive <- function(allowed) {
        allowed <- keep$logical_and(allowed$view(c(1L, 1L, s, s)))
        torch::torch_zeros(c(b, 1L, s, s), dtype = dtype, device = device)$
            masked_fill(allowed$logical_not(), neg)
    }
    list(full_attention = additive(causal),
         sliding_attention = additive(sliding))
}

# Most negative finite value of a float dtype. R torch's torch_finfo()
# rejects bfloat16, whose minimum is -(2 - 2^-7) * 2^127 (float32's
# minimum would round to -Inf there).
.gemma4_dtype_min <- function(dtype) {
    if (identical(dtype$.type(), "BFloat16")) {
        return(-(2 - 2^-7) * 2^127)
    }
    torch::torch_finfo(dtype)$min
}

# Python str.strip(): trims every character str.isspace() accepts, which
# includes Unicode spaces (no-break, ideographic, ...) that base
# trimws() keeps
.py_strip <- function(x) {
    ws <- intToUtf8(c(9:13, 28:32, 0x85, 0xA0, 0x1680, 0x2000:0x200A,
                      0x2028, 0x2029, 0x202F, 0x205F, 0x3000), multiple = TRUE)
    cls <- paste0("[", paste(ws, collapse = ""), "]")
    gsub(paste0("^", cls, "+|", cls, "+$"), "", enc2utf8(x), perl = TRUE)
}

# Model configuration from a checkpoint's text_config (the LTX-2.5 file
# embeds the full HF config as gemma_config metadata)
.gemma4_build_config <- function(raw) {
    tc <- raw$text_config %||% raw
    unsupported <- c(num_kv_shared_layers = 0, hidden_size_per_layer_input = 0)
    for (k in names(unsupported)) {
        if (!is.null(tc[[k]]) && tc[[k]] != unsupported[[k]]) {
            stop("Gemma 4 option ", k, " = ", tc[[k]], " is not supported")
        }
    }
    if (isTRUE(tc$enable_moe_block)) {
        stop("Gemma 4 MoE blocks are not supported")
    }
    list(
         vocab_size = as.integer(tc$vocab_size),
         hidden_size = as.integer(tc$hidden_size),
         intermediate_size = as.integer(tc$intermediate_size),
         num_hidden_layers = as.integer(tc$num_hidden_layers),
         num_attention_heads = as.integer(tc$num_attention_heads),
         num_key_value_heads = as.integer(tc$num_key_value_heads),
         num_global_key_value_heads = as.integer(tc$num_global_key_value_heads %||%
                                                 tc$num_key_value_heads),
         head_dim = as.integer(tc$head_dim),
         global_head_dim = as.integer(tc$global_head_dim %||% 512L),
         attention_k_eq_v = isTRUE(tc$attention_k_eq_v),
         layer_types = as.list(tc$layer_types),
         sliding_window = as.integer(tc$sliding_window),
         rope_parameters = tc$rope_parameters,
         rms_norm_eps = tc$rms_norm_eps %||% 1e-6
    )
}

# Checkpoint key -> module name: text weights only (model.* in the LTX
# file, model.language_model.* in a transformers export); NA skips the
# vision/audio towers, projectors, the connectors' text projection and
# the embedded tokenizer assets
.gemma4_map_key <- function(key) {
    if (grepl("^model\\.language_model\\.", key)) {
        return(sub("^model\\.language_model\\.", "", key))
    }
    if (grepl("^model\\.(embed_tokens|layers|norm)\\.", key)) {
        return(sub("^model\\.", "", key))
    }
    NA_character_
}

#' Load the Gemma 4 text encoder (LTX-2.5)
#'
#' Streams the text weights of an LTX-2.5 Gemma 4 checkpoint into a
#' \code{\link{gemma4_text_model}} skeleton, one tensor at a time, so
#' the 26 GB bf16 file is never fully in memory. The vision and audio
#' towers in the same file are skipped.
#'
#' @param path Path to the LTX-2.5 text encoder file (e.g.
#'   \code{gemma4-12b-with-proj-ltx-2.5-bf16.safetensors}), which embeds
#'   its config as \code{gemma_config} metadata, or an NF4 artifact
#'   directory from \code{\link{gemma4_quantize_nf4}} (dispatched to
#'   \code{\link{load_gemma4_nf4}}).
#' @param device Character. Device for the loaded model.
#' @param dtype Character. "bfloat16" (checkpoint native) or "float32".
#' @param pin Logical. When loading to the CPU, page-lock the weights so
#'   \code{\link{encode_with_gemma4}} can move the model to the GPU per
#'   encode at DMA speed. Default follows
#'   \code{options(diffuseR.pin_staging)}.
#' @param stream Logical. Stage layer by layer (see
#'   \code{\link{load_gemma4_nf4}}).
#' @param verbose Logical.
#'
#' @return A \code{gemma4_text_model}.
#'
#' @export
load_gemma4_text_encoder <- function(path, device = "cpu", dtype = "bfloat16",
                                     pin = getOption("diffuseR.pin_staging", TRUE),
                                     stream = FALSE, verbose = TRUE) {
    if (file.exists(file.path(path, "manifest.json"))) {
        return(load_gemma4_nf4(path, device = device, dtype = dtype,
                               pin = pin, stream = stream, verbose = verbose))
    }
    ckpt <- ltx23_open_checkpoint(path, require_version = NULL)
    meta <- ckpt$handle$metadata[["__metadata__"]]
    if (is.null(meta[["gemma_config"]])) {
        stop("No gemma_config metadata in ", path,
             "; expected an LTX-2.5 Gemma 4 text encoder file")
    }
    config <- .gemma4_build_config(jsonlite::fromJSON(meta[["gemma_config"]],
                                                      simplifyVector = FALSE))
    torch_dtype <- switch(dtype, bfloat16 = torch::torch_bfloat16(),
                          float32 = torch::torch_float32(),
                          stop("Unsupported dtype: ", dtype))
    if (verbose) {
        message(sprintf("Creating Gemma 4 model: %d layers, hidden_size=%d",
                        config$num_hidden_layers, config$hidden_size))
    }
    model <- .construct_skeleton(gemma4_text_model, config, dtype = torch_dtype)

    keys <- ckpt$keys
    res <- ltx23_load_group(ckpt, keys, model, map_key = .gemma4_map_key,
                            verbose = verbose)
    if (length(res$unmapped) || length(res$unfilled)) {
        stop("Gemma 4 load: ", length(res$unmapped), " unmapped, ",
             length(res$unfilled), " unfilled (e.g. ",
             paste(utils::head(c(res$unmapped, res$unfilled), 3), collapse = ", "),
             ")")
    }
    model$to(device = device)
    model$eval()

    if (stream && device == "cpu") {
        attr(model, "staging") <- .stage_layer_stream(model, model$layers,
                                                      "layers.", pin)
    } else if (pin && device == "cpu") {
        st <- .pin_component(model)
        if (!is.null(st)) {
            attr(model, "staging") <- st
            if (verbose) {
                message("Gemma 4 weights pinned for staged transfer")
            }
        }
    }
    model
}

#' Encode prompts with Gemma 4 for LTX-2.5
#'
#' The LTX-2.5 text-encoder step: strips each prompt (Python
#' \code{str.strip()} semantics, Unicode spaces included), tokenizes with left
#' padding to \code{max_sequence_length}, and returns every hidden state
#' (embedding output plus each layer, the last final-normed) stacked on a
#' new last dimension for the connectors.
#'
#' @param prompts Character vector.
#' @param model A \code{\link{gemma4_text_model}}. A pinned CPU-resident
#'   model moves to \code{device} for the encode and back afterwards.
#' @param tokenizer A \code{\link{gemma4_tokenizer}}.
#' @param max_sequence_length Integer. Padded token length.
#' @param device Character. Compute device.
#' @param verbose Logical.
#'
#' @return List with \code{prompt_embeds} \code{[batch, seq_len,
#'   hidden_size, num_layers + 1]} and \code{prompt_attention_mask}
#'   \code{[batch, seq_len]}, the same layout as
#'   \code{\link{encode_with_gemma3}}.
#'
#' @export
encode_with_gemma4 <- function(prompts, model, tokenizer,
                               max_sequence_length = 1024L, device = "cuda",
                               verbose = TRUE) {
    stopifnot(inherits(tokenizer, "gemma4_tokenizer"))
    # Same staging contract as encode_with_gemma3(): arm the offload
    # before the onload so a partial onload is still undone
    staging <- attr(model, "staging")
    if (!is.null(staging) && device != "cpu" && !.staged_on(staging, device)) {
        on.exit(.staged_offload(staging), add = TRUE)
        .staged_onload(staging, device)
    }
    tokens <- encode_gemma4(tokenizer, .py_strip(as.character(prompts)),
                            max_length = max_sequence_length)
    input_ids <- torch::torch_tensor(tokens$input_ids, dtype = torch::torch_long(),
                                     device = device)
    attention_mask <- torch::torch_tensor(tokens$attention_mask,
                                          dtype = torch::torch_long(),
                                          device = device)
    if (verbose) {
        message("Encoding with Gemma 4...")
    }
    torch::with_no_grad({
        out <- model(input_ids, attention_mask = attention_mask,
                     output_hidden_states = TRUE)
    })
    list(prompt_embeds = torch::torch_stack(out$hidden_states, dim = -1L),
         prompt_attention_mask = attention_mask)
}
