#' LTX-2.3 Transformer Building Blocks
#'
#' Fresh R port of the LTX-2 transformer components from the diffusers
#' reference (Apache-2.0, src/diffusers/models/transformers/
#' transformer_ltx2.py and shared modules). Field names mirror the
#' diffusers module tree so checkpoint keys map 1:1.
#'
#' @name dit_ltx23_modules
NULL

#' RMS normalization
#'
#' Variance is computed in float32; the result is cast back to the input
#' dtype (or the weight dtype when elementwise affine).
#'
#' @param dim Integer. Normalized dimension size.
#' @param eps Numeric. Stability epsilon.
#' @param elementwise_affine Logical. Learn a scale weight.
#'
#' @export
ltx23_rms_norm <- torch::nn_module(
                                   "ltx23_rms_norm",
                                   initialize = function(dim, eps = 1e-6, elementwise_affine = TRUE) {
    self$eps <- eps
    self$elementwise_affine <- elementwise_affine
    if (elementwise_affine) {
        self$weight <- torch::nn_parameter(torch::torch_ones(dim))
    }
},
                                   forward = function(x) {
    input_dtype <- x$dtype
    variance <- x$to(dtype = torch::torch_float32())$pow(2)$mean(dim = -1L,
        keepdim = TRUE)
    x <- x * torch::torch_rsqrt(variance + self$eps)
    if (self$elementwise_affine) {
        x <- x$to(dtype = self$weight$dtype) * self$weight
    } else {
        x <- x$to(dtype = input_dtype)
    }
    x
}
)

#' Sinusoidal timestep embedding
#'
#' DDPM-style sinusoidal embedding. LTX uses \code{flip_sin_to_cos=TRUE}
#' (cos first) and \code{downscale_freq_shift=0}.
#'
#' @param timesteps 1D tensor of timestep values.
#' @param embedding_dim Integer. Output embedding size.
#' @param flip_sin_to_cos Logical. Put cos before sin.
#' @param downscale_freq_shift Numeric. Frequency delta control.
#' @param max_period Numeric. Maximum embedding frequency period.
#'
#' @return Tensor [N, embedding_dim].
#'
#' @export
ltx23_get_timestep_embedding <- function(timesteps, embedding_dim,
    flip_sin_to_cos = TRUE,
    downscale_freq_shift = 0,
    max_period = 10000) {
    stopifnot(timesteps$ndim == 1L)
    half_dim <- embedding_dim %/% 2L
    exponent <- -log(max_period) * torch::torch_arange(start = 0,
        end = half_dim - 1, dtype = torch::torch_float32(),
        device = timesteps$device)
    exponent <- exponent / (half_dim - downscale_freq_shift)

    emb <- torch::torch_exp(exponent)
    emb <- timesteps$unsqueeze(2L)$to(dtype = torch::torch_float32()) * emb$unsqueeze(1L)
    emb <- torch::torch_cat(list(torch::torch_sin(emb), torch::torch_cos(emb)), dim = -1L)

    if (flip_sin_to_cos) {
        emb <- torch::torch_cat(
                                list(emb[, (half_dim + 1):(2 * half_dim)], emb[, 1:half_dim]),
                                dim = -1L
        )
    }
    if (embedding_dim %% 2L == 1L) {
        emb <- torch::nnf_pad(emb, c(0L, 1L, 0L, 0L))
    }
    emb
}

# Two-layer timestep MLP (linear_1 -> silu -> linear_2), matching
# diffusers TimestepEmbedding state-dict names.
ltx23_timestep_embedding <- torch::nn_module(
    "ltx23_timestep_embedding",
    initialize = function(in_channels, time_embed_dim, bias = TRUE) {
    self$linear_1 <- torch::nn_linear(in_channels, time_embed_dim, bias = bias)
    self$linear_2 <- torch::nn_linear(time_embed_dim, time_embed_dim,
                                      bias = bias)
},
    forward = function(sample) {
    self$linear_2(torch::nnf_silu(self$linear_1(sample)))
}
)

# Sinusoidal projection (256 channels) + MLP, matching diffusers
# PixArtAlphaCombinedTimestepSizeEmbeddings without additional conditions.
ltx23_combined_timestep_embed <- torch::nn_module(
    "ltx23_combined_timestep_embed",
    initialize = function(embedding_dim) {
    self$timestep_embedder <- ltx23_timestep_embedding(256L, embedding_dim)
},
    forward = function(timestep, hidden_dtype) {
    proj <- ltx23_get_timestep_embedding(timestep, 256L,
        flip_sin_to_cos = TRUE, downscale_freq_shift = 0)
    self$timestep_embedder(proj$to(dtype = hidden_dtype))
}
)

#' Adaptive layer norm single (adaLN-single)
#'
#' Embeds a timestep/sigma and projects it to a configurable number of
#' modulation parameter vectors.
#'
#' @param embedding_dim Integer. Model dimension.
#' @param num_mod_params Integer. Number of modulation parameter vectors.
#'
#' @return Module whose forward returns
#'   \code{list(mod_params [N, num_mod_params * dim], embedded_timestep [N, dim])}.
#'
#' @export
ltx23_ada_layer_norm_single <- torch::nn_module(
    "ltx23_ada_layer_norm_single",
    initialize = function(embedding_dim, num_mod_params = 6L) {
    self$num_mod_params <- num_mod_params
    self$emb <- ltx23_combined_timestep_embed(embedding_dim)
    self$linear <- torch::nn_linear(embedding_dim,
                                    num_mod_params * embedding_dim,
                                    bias = TRUE)
},
    forward = function(timestep, hidden_dtype) {
    embedded_timestep <- self$emb(timestep, hidden_dtype = hidden_dtype)
    list(self$linear(torch::nnf_silu(embedded_timestep)), embedded_timestep)
}
)

# torch_matmul_out / torch_softmax_out are unexported torch internals;
# resolve them once and fall back to allocating ops if they disappear
.ltx23_attn_out_fns <- local({
    fns <- NULL
    function() {
        if (is.null(fns)) {
            fns <<- tryCatch(
                             list(
                                  matmul = get("torch_matmul_out",
                        envir = asNamespace("torch")),
                                  softmax = get("torch_softmax_out", envir = asNamespace("torch"))
                ),
                             error = function(e) FALSE
            )
        }
        fns
    }
})

# Persistent attention scratch (score matrix + context output), keyed by
# shape/dtype/device like the dequant buffers: attention temporaries are
# the dominant per-block garbage at high resolution
.ltx23_attn_scratch <- new.env(parent = emptyenv())

.ltx23_get_attn_buffer <- function(shape, dtype, device) {
    key <- paste(paste(shape, collapse = "x"), dtype$.type(),
                 paste(device$type, device$index %||% 0L), sep = "|")
    buf <- .ltx23_attn_scratch[[key]]
    if (is.null(buf)) {
        buf <- torch::torch_empty(shape, dtype = dtype, device = device)
        .ltx23_attn_scratch[[key]] <- buf
    }
    buf
}

# Scaled dot-product attention on [B, H, S, D] tensors. R torch has no
# fused SDPA, so the [B, H, Sq, Sk] score matrix materializes; queries
# are chunked adaptively against a memory budget and all large
# temporaries live in reusable scratch buffers.
.ltx23_sdpa <- function(query, key, value, attention_mask = NULL,
                        chunk_size = NULL) {
    head_dim <- query$shape[length(query$shape)]
    scale <- 1.0 / sqrt(head_dim)
    key_t <- key$transpose(-2L, -1L)
    fns <- .ltx23_attn_out_fns()

    b <- query$shape[1]
    heads <- query$shape[2]
    n_q <- query$shape[3]
    n_k <- key$shape[3]
    d_v <- value$shape[4]

    attend <- function(q, mask) {
        attn <- torch::torch_matmul(q$mul(scale), key_t)
        if (!is.null(mask)) {
            attn <- attn + mask
        }
        attn <- torch::nnf_softmax(attn, dim = -1L)
        torch::torch_matmul(attn, value)
    }

    # Buffered variant: score matrix and context reuse persistent scratch
    attend_into <- function(q, mask, attn_buf, out_buf) {
        fns$matmul(attn_buf, q$mul(scale), key_t)
        if (!is.null(mask)) {
            attn_buf$add_(mask)
        }
        fns$softmax(attn_buf, attn_buf, dim = -1L, dtype = attn_buf$dtype)
        fns$matmul(out_buf, attn_buf, value)
        out_buf
    }

    # Adaptive query chunking: bound the [B, H, chunk, S_k] score matrix
    # to a memory budget regardless of sequence length. The default is
    # sized so the self+cross scratch pair stays under ~400MB at
    # 1280x704x121 (14080 video tokens), the validated ceiling of the
    # 16GB NF4-resident profile; below ~2000 tokens it has no effect.
    budget <- getOption("diffuseR.attn_budget", 1.5e8)
    auto_chunk <- max(256L, as.integer(budget / (b * heads * n_k * 2)))
    if (is.null(chunk_size)) {
        chunk_size <- auto_chunk
    } else {
        chunk_size <- min(chunk_size, auto_chunk)
    }

    if (isFALSE(fns)) {
        # Fallback: allocating path
        if (n_q <= chunk_size) {
            return(attend(query, attention_mask))
        }
        outs <- list()
        start <- 1L
        while (start <= n_q) {
            len <- min(chunk_size, n_q - start + 1L)
            q_chunk <- query$narrow(3L, start, len)
            mask_chunk <- attention_mask
            if (!is.null(mask_chunk) && mask_chunk$shape[3] > 1L) {
                mask_chunk <- mask_chunk$narrow(3L, start, len)
            }
            outs[[length(outs) + 1L]] <- attend(q_chunk, mask_chunk)
            start <- start + len
        }
        return(torch::torch_cat(outs, dim = 3L))
    }

    rows <- min(n_q, chunk_size)
    attn_buf <- .ltx23_get_attn_buffer(c(b, heads, rows, n_k), query$dtype,
                                       query$device)
    out_buf <- .ltx23_get_attn_buffer(c(b, heads, n_q, d_v), query$dtype,
                                      query$device)

    if (n_q <= chunk_size) {
        return(attend_into(query, attention_mask, attn_buf, out_buf))
    }

    start <- 1L
    while (start <= n_q) {
        len <- min(chunk_size, n_q - start + 1L)
        q_chunk <- query$narrow(3L, start, len)
        mask_chunk <- attention_mask
        if (!is.null(mask_chunk) && mask_chunk$shape[3] > 1L) {
            mask_chunk <- mask_chunk$narrow(3L, start, len)
        }
        attend_into(
                    q_chunk, mask_chunk,
                    attn_buf$narrow(3L, 1L, len),
                    out_buf$narrow(3L, start, len)
        )
        start <- start + len
    }
    out_buf
}

#' Release the attention scratch buffers
#'
#' @return Invisibly, NULL.
#'
#' @keywords internal
.ltx23_release_attn_buffers <- function() {
    rm(list = ls(.ltx23_attn_scratch), envir = .ltx23_attn_scratch)
    invisible(NULL)
}

#' LTX-2 attention layer
#'
#' Attention with RMS q/k norms across heads, optional per-head output
#' gating (LTX-2.3), separate query/key RoPE (for a2v/v2a cross
#' attention), and optional STG perturbation (skip attention, use the
#' value projection).
#'
#' @param query_dim Integer. Query feature dimension.
#' @param heads,kv_heads Integers. Attention head counts.
#' @param dim_head Integer. Per-head dimension.
#' @param bias,out_bias Logicals. Projection biases.
#' @param cross_attention_dim Integer or NULL. Key/value input dimension
#'   (NULL for self-attention).
#' @param norm_eps Numeric. RMS norm epsilon.
#' @param norm_elementwise_affine Logical. RMS norms carry weights.
#' @param rope_type "split" or "interleaved".
#' @param apply_gated_attention Logical. Add per-head sigmoid output gates.
#'
#' @export
ltx23_attention <- torch::nn_module(
                                    "ltx23_attention",
                                    initialize = function(
        query_dim,
        heads = 8L,
        kv_heads = NULL,
        dim_head = 64L,
        bias = TRUE,
        cross_attention_dim = NULL,
        out_bias = TRUE,
        norm_eps = 1e-6,
        norm_elementwise_affine = TRUE,
        rope_type = "split",
        apply_gated_attention = FALSE
    ) {
    kv_heads <- kv_heads %||% heads
    self$heads <- as.integer(heads)
    self$head_dim <- as.integer(dim_head)
    inner_dim <- dim_head * heads
    inner_kv_dim <- dim_head * kv_heads
    cross_dim <- cross_attention_dim %||% query_dim
    self$rope_type <- rope_type
    self$attn_chunk <- NULL # set by memory profiles; NULL = unchunked

    self$norm_q <- ltx23_rms_norm(inner_dim, eps = norm_eps,
                                  elementwise_affine = norm_elementwise_affine)
    self$norm_k <- ltx23_rms_norm(inner_kv_dim, eps = norm_eps,
                                  elementwise_affine = norm_elementwise_affine)
    self$to_q <- torch::nn_linear(query_dim, inner_dim, bias = bias)
    self$to_k <- torch::nn_linear(cross_dim, inner_kv_dim, bias = bias)
    self$to_v <- torch::nn_linear(cross_dim, inner_kv_dim, bias = bias)
    self$to_out <- torch::nn_module_list(list(
            torch::nn_linear(inner_dim, query_dim, bias = out_bias)
        ))

    self$apply_gated_attention <- apply_gated_attention
    if (apply_gated_attention) {
        # Per-head gate logits, computed on the raw block input (pre-Q)
        self$to_gate_logits <- torch::nn_linear(query_dim, heads, bias = TRUE)
    }
},
                                    forward = function(
        hidden_states,
        encoder_hidden_states = NULL,
        attention_mask = NULL,
        query_rotary_emb = NULL,
        key_rotary_emb = NULL,
        perturbation_mask = NULL,
        all_perturbed = FALSE
    ) {
    if (is.null(encoder_hidden_states)) {
        encoder_hidden_states <- hidden_states
    }
    if (!is.null(attention_mask)) {
        # [B, 1, S] or [B, Tq, S] additive bias -> [B, 1, *, S]
        if (attention_mask$ndim == 3L) {
            attention_mask <- attention_mask$unsqueeze(2L)
        }
    }

    gate_logits <- NULL
    if (self$apply_gated_attention) {
        gate_logits <- self$to_gate_logits(hidden_states) # [B, T, H]
    }

    value_flat <- self$to_v(encoder_hidden_states)

    if (isTRUE(all_perturbed)) {
        # STG: skip attention entirely, use the value projection
        out <- value_flat
    } else {
        query <- self$norm_q(self$to_q(hidden_states))
        key <- self$norm_k(self$to_k(encoder_hidden_states))

        key_rope <- key_rotary_emb %||% query_rotary_emb
        if (!is.null(query_rotary_emb)) {
            if (self$rope_type == "interleaved") {
                query <- ltx23_apply_interleaved_rotary_emb(query, query_rotary_emb)
                key <- ltx23_apply_interleaved_rotary_emb(key, key_rope)
            } else {
                query <- ltx23_apply_split_rotary_emb(query, query_rotary_emb)
                key <- ltx23_apply_split_rotary_emb(key, key_rope)
            }
        }

        # [B, S, H*D] -> [B, H, S, D]
        query <- query$unflatten(3L, c(self$heads, -1L))$transpose(2L, 3L)
        key <- key$unflatten(3L, c(self$heads, -1L))$transpose(2L, 3L)
        value <- value_flat$unflatten(3L, c(self$heads, -1L))$transpose(2L, 3L)

        out <- .ltx23_sdpa(query, key, value, attention_mask, self$attn_chunk)
        out <- out$transpose(2L, 3L)$flatten(start_dim = 3L)
        out <- out$to(dtype = hidden_states$dtype)

        if (!is.null(perturbation_mask)) {
            # Interpolate between the perturbed (value) and full attention paths
            out <- torch::torch_lerp(value_flat, out, perturbation_mask)
        }
    }

    if (self$apply_gated_attention) {
        out <- out$unflatten(3L, c(self$heads, -1L)) # [B, T, H, D]
        # Factor 2 so zero-initialized gate logits give unit gates
        gates <- gate_logits$sigmoid()$mul(2.0) # [B, T, H]
        out <- out * gates$unsqueeze(-1L)
        out <- out$flatten(start_dim = 3L)
    }

    self$to_out[[1]](out)
}
)

# In-place tanh GELU for large activations: nnf_gelu allocates a second
# copy of its input, which at high resolution doubles the feed-forward
# intermediate (the single largest transient in a block). Uses the
# internal in-place kernel when available, chunked so any internal
# temporaries stay small; falls back to the allocating path.
.ltx23_gelu_tanh_inplace <- function(x, chunk_elements = 2 ^ 24) {
    fn <- get0("torch_gelu_", envir = asNamespace("torch"))
    if (is.null(fn)) {
        return(torch::nnf_gelu(x, approximate = "tanh"))
    }
    flat <- x$view(-1L)
    n <- flat$numel()
    start <- 1
    while (start <= n) {
        len <- min(chunk_elements, n - start + 1)
        fn(flat$narrow(1L, start, len), approximate = "tanh")
        start <- start + len
    }
    x
}

#' LTX feed-forward layer
#'
#' Linear -> GELU (tanh approximation) -> Linear with 4x hidden dim,
#' matching diffusers \code{FeedForward(activation_fn="gelu-approximate")}
#' state-dict names (net.0.proj, net.2).
#'
#' @param dim Integer. Input/output dimension.
#' @param mult Integer. Hidden dimension multiplier.
#'
#' @export
ltx23_feed_forward <- torch::nn_module(
                                       "ltx23_feed_forward",
                                       initialize = function(dim, mult = 4L) {
    inner_dim <- as.integer(dim * mult)
    gelu_proj <- torch::nn_module(
                                  "ltx23_gelu",
                                  initialize = function(dim_in, dim_out) {
        self$proj <- torch::nn_linear(dim_in, dim_out)
    },
                                  forward = function(x) {
        h <- self$proj(x)
        if (h$numel() > getOption("diffuseR.gelu_inplace_min", 1e8)) {
            return(.ltx23_gelu_tanh_inplace(h))
        }
        torch::nnf_gelu(h, approximate = "tanh")
    }
    )
    self$net <- torch::nn_module_list(list(
            gelu_proj(dim, inner_dim),
            torch::nn_identity(), # dropout slot in the reference; inference no-op
            torch::nn_linear(inner_dim, dim)
        ))
},
                                       forward = function(x) {
    self$net[[3]](self$net[[1]](x))
}
)

# Combine a per-block scale/shift table with global modulation params:
# ada_values = table[None, None] + temb.reshape(B, temb_tokens, num, -1),
# returned as a list split along the parameter axis.
ltx23_get_mod_params <- function(scale_shift_table, temb, batch_size) {
    num_params <- scale_shift_table$shape[1]
    ada_values <- scale_shift_table$unsqueeze(1L)$unsqueeze(1L)$to(device = temb$device) +
    temb$reshape(c(batch_size, temb$shape[2], num_params, -1L))
    torch::torch_unbind(ada_values, dim = 3L)
}

#' LTX-2 transformer block
#'
#' Dual-stream (video + audio) block: modulated self-attention per
#' modality, text cross-attention per modality (with LTX-2.3 query and
#' key/value modulation), bidirectional audio-video cross-attention with
#' global+per-block modulation, and modulated feed-forward.
#'
#' @param dim,audio_dim Integers. Video/audio stream dimensions.
#' @param num_attention_heads,attention_head_dim Video attention shape.
#' @param cross_attention_dim Integer. Text embedding dim for video.
#' @param audio_num_attention_heads,audio_attention_head_dim Audio attention shape.
#' @param audio_cross_attention_dim Integer. Text embedding dim for audio.
#' @param video_gated_attn,audio_gated_attn Logicals. Per-head output gates.
#' @param video_cross_attn_adaln,audio_cross_attn_adaln Logicals. LTX-2.3
#'   text cross-attention modulation (9 mod params instead of 6).
#' @param eps Numeric. Norm epsilon.
#' @param elementwise_affine Logical. Block norms carry weights (FALSE for LTX).
#' @param rope_type "split" or "interleaved".
#' @param perturbed_attn Logical. Enable the STG perturbation arguments.
#'
#' @export
ltx23_transformer_block <- torch::nn_module(
    "ltx23_transformer_block",
    initialize = function(
                          dim,
                          num_attention_heads,
                          attention_head_dim,
                          cross_attention_dim,
                          audio_dim,
                          audio_num_attention_heads,
                          audio_attention_head_dim,
                          audio_cross_attention_dim,
                          video_gated_attn = TRUE,
                          video_cross_attn_adaln = TRUE,
                          audio_gated_attn = TRUE,
                          audio_cross_attn_adaln = TRUE,
                          eps = 1e-6,
                          elementwise_affine = FALSE,
                          rope_type = "split",
                          perturbed_attn = TRUE
    ) {
    self$perturbed_attn <- perturbed_attn

    # 1. Self-attention (video and audio)
    self$norm1 <- ltx23_rms_norm(dim, eps = eps,
                                 elementwise_affine = elementwise_affine)
    self$attn1 <- ltx23_attention(
                                  query_dim = dim, heads = num_attention_heads, dim_head = attention_head_dim,
                                  rope_type = rope_type, apply_gated_attention = video_gated_attn
    )
    self$audio_norm1 <- ltx23_rms_norm(audio_dim, eps = eps, elementwise_affine = elementwise_affine)
    self$audio_attn1 <- ltx23_attention(
                                        query_dim = audio_dim, heads = audio_num_attention_heads,
                                        dim_head = audio_attention_head_dim,
                                        rope_type = rope_type, apply_gated_attention = audio_gated_attn
    )

    # 2. Prompt cross-attention
    self$norm2 <- ltx23_rms_norm(dim, eps = eps, elementwise_affine = elementwise_affine)
    self$attn2 <- ltx23_attention(
                                  query_dim = dim, cross_attention_dim = cross_attention_dim,
                                  heads = num_attention_heads, dim_head = attention_head_dim,
                                  rope_type = rope_type, apply_gated_attention = video_gated_attn
    )
    self$audio_norm2 <- ltx23_rms_norm(audio_dim, eps = eps, elementwise_affine = elementwise_affine)
    self$audio_attn2 <- ltx23_attention(
                                        query_dim = audio_dim, cross_attention_dim = audio_cross_attention_dim,
                                        heads = audio_num_attention_heads, dim_head = audio_attention_head_dim,
                                        rope_type = rope_type, apply_gated_attention = audio_gated_attn
    )

    # 3. Audio-to-video (Q: video) and video-to-audio (Q: audio) cross-attention
    self$audio_to_video_norm <- ltx23_rms_norm(dim, eps = eps, elementwise_affine = elementwise_affine)
    self$audio_to_video_attn <- ltx23_attention(
        query_dim = dim, cross_attention_dim = audio_dim,
        heads = audio_num_attention_heads, dim_head = audio_attention_head_dim,
        rope_type = rope_type, apply_gated_attention = video_gated_attn
    )
    self$video_to_audio_norm <- ltx23_rms_norm(audio_dim, eps = eps, elementwise_affine = elementwise_affine)
    self$video_to_audio_attn <- ltx23_attention(
        query_dim = audio_dim, cross_attention_dim = dim,
        heads = audio_num_attention_heads, dim_head = audio_attention_head_dim,
        rope_type = rope_type, apply_gated_attention = audio_gated_attn
    )

    # 4. Feed-forward
    self$norm3 <- ltx23_rms_norm(dim, eps = eps, elementwise_affine = elementwise_affine)
    self$ff <- ltx23_feed_forward(dim)
    self$audio_norm3 <- ltx23_rms_norm(audio_dim, eps = eps, elementwise_affine = elementwise_affine)
    self$audio_ff <- ltx23_feed_forward(audio_dim)

    # 5. Per-block modulation tables
    self$video_cross_attn_adaln <- video_cross_attn_adaln
    self$audio_cross_attn_adaln <- audio_cross_attn_adaln
    self$cross_attn_adaln <- video_cross_attn_adaln || audio_cross_attn_adaln
    video_mod_params <- if (video_cross_attn_adaln) 9L else 6L
    audio_mod_params <- if (audio_cross_attn_adaln) 9L else 6L
    self$scale_shift_table <- torch::nn_parameter(
        torch::torch_randn(video_mod_params, dim) / sqrt(dim)
    )
    self$audio_scale_shift_table <- torch::nn_parameter(
        torch::torch_randn(audio_mod_params, audio_dim) / sqrt(audio_dim)
    )
    if (self$cross_attn_adaln) {
        self$prompt_scale_shift_table <- torch::nn_parameter(torch::torch_randn(2L, dim))
        self$audio_prompt_scale_shift_table <- torch::nn_parameter(torch::torch_randn(2L, audio_dim))
    }
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
                       temb_prompt = NULL,
                       temb_prompt_audio = NULL,
                       video_rotary_emb = NULL,
                       audio_rotary_emb = NULL,
                       ca_video_rotary_emb = NULL,
                       ca_audio_rotary_emb = NULL,
                       encoder_attention_mask = NULL,
                       audio_encoder_attention_mask = NULL,
                       self_attention_mask = NULL,
                       use_a2v_cross_attention = TRUE,
                       use_v2a_cross_attention = TRUE,
                       perturbation_mask = NULL,
                       all_perturbed = FALSE
    ) {
    batch_size <- hidden_states$shape[1]

    # 1.1 Video self-attention
    video_ada <- ltx23_get_mod_params(self$scale_shift_table, temb, batch_size)
    shift_msa <- video_ada[[1]]; scale_msa <- video_ada[[2]]; gate_msa <- video_ada[[3]]
    shift_mlp <- video_ada[[4]]; scale_mlp <- video_ada[[5]]; gate_mlp <- video_ada[[6]]

    norm_hidden_states <- self$norm1(hidden_states)
    norm_hidden_states <- norm_hidden_states * scale_msa$add(1) + shift_msa

    attn_hidden_states <- self$attn1(
                                     norm_hidden_states,
                                     query_rotary_emb = video_rotary_emb,
                                     attention_mask = self_attention_mask,
                                     perturbation_mask = if (self$perturbed_attn) {
            perturbation_mask
        } else {
            NULL
        },
                                     all_perturbed = if (self$perturbed_attn) all_perturbed else FALSE
    )
    hidden_states <- hidden_states + attn_hidden_states * gate_msa

    # 1.2 Audio self-attention
    audio_ada <- ltx23_get_mod_params(self$audio_scale_shift_table, temb_audio, batch_size)
    audio_shift_msa <- audio_ada[[1]]; audio_scale_msa <- audio_ada[[2]]
    audio_gate_msa <- audio_ada[[3]]; audio_shift_mlp <- audio_ada[[4]]
    audio_scale_mlp <- audio_ada[[5]]; audio_gate_mlp <- audio_ada[[6]]

    norm_audio_hidden_states <- self$audio_norm1(audio_hidden_states)
    norm_audio_hidden_states <- norm_audio_hidden_states * audio_scale_msa$add(1) + audio_shift_msa

    attn_audio_hidden_states <- self$audio_attn1(
        norm_audio_hidden_states,
        query_rotary_emb = audio_rotary_emb,
        perturbation_mask = if (self$perturbed_attn) {
            perturbation_mask
        } else {
            NULL
        },
        all_perturbed = if (self$perturbed_attn) all_perturbed else FALSE
    )
    audio_hidden_states <- audio_hidden_states + attn_audio_hidden_states * audio_gate_msa

    # 2. Text cross-attention modulation params (LTX-2.3)
    if (self$cross_attn_adaln) {
        prompt_ada <- ltx23_get_mod_params(self$prompt_scale_shift_table, temb_prompt, batch_size)
        shift_text_kv <- prompt_ada[[1]]; scale_text_kv <- prompt_ada[[2]]
        audio_prompt_ada <- ltx23_get_mod_params(
            self$audio_prompt_scale_shift_table, temb_prompt_audio, batch_size
        )
        audio_shift_text_kv <- audio_prompt_ada[[1]]; audio_scale_text_kv <- audio_prompt_ada[[2]]
    }

    # 2.1 Video-text cross-attention
    norm_hidden_states <- self$norm2(hidden_states)
    if (self$video_cross_attn_adaln) {
        shift_text_q <- video_ada[[7]]; scale_text_q <- video_ada[[8]]; gate_text_q <- video_ada[[9]]
        norm_hidden_states <- norm_hidden_states * scale_text_q$add(1) + shift_text_q
    }
    enc_states <- encoder_hidden_states
    if (self$cross_attn_adaln) {
        enc_states <- enc_states * scale_text_kv$add(1) + shift_text_kv
    }
    attn_hidden_states <- self$attn2(
                                     norm_hidden_states,
                                     encoder_hidden_states = enc_states,
                                     attention_mask = encoder_attention_mask
    )
    if (self$video_cross_attn_adaln) {
        attn_hidden_states <- attn_hidden_states * gate_text_q
    }
    hidden_states <- hidden_states + attn_hidden_states

    # 2.2 Audio-text cross-attention
    norm_audio_hidden_states <- self$audio_norm2(audio_hidden_states)
    if (self$audio_cross_attn_adaln) {
        audio_shift_text_q <- audio_ada[[7]]; audio_scale_text_q <- audio_ada[[8]]
        audio_gate_text_q <- audio_ada[[9]]
        norm_audio_hidden_states <- norm_audio_hidden_states * audio_scale_text_q$add(1) +
        audio_shift_text_q
    }
    audio_enc_states <- audio_encoder_hidden_states
    if (self$cross_attn_adaln) {
        audio_enc_states <- audio_enc_states * audio_scale_text_kv$add(1) + audio_shift_text_kv
    }
    attn_audio_hidden_states <- self$audio_attn2(
        norm_audio_hidden_states,
        encoder_hidden_states = audio_enc_states,
        attention_mask = audio_encoder_attention_mask
    )
    if (self$audio_cross_attn_adaln) {
        attn_audio_hidden_states <- attn_audio_hidden_states * audio_gate_text_q
    }
    audio_hidden_states <- audio_hidden_states + attn_audio_hidden_states

    # 3. Audio-to-video and video-to-audio cross-attention
    if (use_a2v_cross_attention || use_v2a_cross_attention) {
        norm_hidden_states <- self$audio_to_video_norm(hidden_states)
        norm_audio_hidden_states <- self$video_to_audio_norm(audio_hidden_states)

        video_ca_ada <- ltx23_get_mod_params(
            self$video_a2v_cross_attn_scale_shift_table$narrow(1L, 1L, 4L),
            temb_ca_scale_shift, batch_size
        )
        video_ca_gate <- ltx23_get_mod_params(
            self$video_a2v_cross_attn_scale_shift_table$narrow(1L, 5L, 1L),
            temb_ca_gate, batch_size
        )
        a2v_gate <- video_ca_gate[[1]]$squeeze(3L)

        audio_ca_ada <- ltx23_get_mod_params(
            self$audio_a2v_cross_attn_scale_shift_table$narrow(1L, 1L, 4L),
            temb_ca_audio_scale_shift, batch_size
        )
        audio_ca_gate <- ltx23_get_mod_params(
            self$audio_a2v_cross_attn_scale_shift_table$narrow(1L, 5L, 1L),
            temb_ca_audio_gate, batch_size
        )
        v2a_gate <- audio_ca_gate[[1]]$squeeze(3L)

        if (use_a2v_cross_attention) {
            mod_norm_hidden <- norm_hidden_states *
            video_ca_ada[[1]]$squeeze(3L)$add(1) + video_ca_ada[[2]]$squeeze(3L)
            mod_norm_audio <- norm_audio_hidden_states *
            audio_ca_ada[[1]]$squeeze(3L)$add(1) + audio_ca_ada[[2]]$squeeze(3L)

            a2v_attn <- self$audio_to_video_attn(
                mod_norm_hidden,
                encoder_hidden_states = mod_norm_audio,
                query_rotary_emb = ca_video_rotary_emb,
                key_rotary_emb = ca_audio_rotary_emb
            )
            hidden_states <- hidden_states + a2v_gate * a2v_attn
        }

        if (use_v2a_cross_attention) {
            mod_norm_hidden <- norm_hidden_states *
            video_ca_ada[[3]]$squeeze(3L)$add(1) + video_ca_ada[[4]]$squeeze(3L)
            mod_norm_audio <- norm_audio_hidden_states *
            audio_ca_ada[[3]]$squeeze(3L)$add(1) + audio_ca_ada[[4]]$squeeze(3L)

            v2a_attn <- self$video_to_audio_attn(
                mod_norm_audio,
                encoder_hidden_states = mod_norm_hidden,
                query_rotary_emb = ca_audio_rotary_emb,
                key_rotary_emb = ca_video_rotary_emb
            )
            audio_hidden_states <- audio_hidden_states + v2a_gate * v2a_attn
        }
    }

    # 4. Feed-forward
    norm_hidden_states <- self$norm3(hidden_states) * scale_mlp$add(1) + shift_mlp
    hidden_states <- hidden_states + self$ff(norm_hidden_states) * gate_mlp

    norm_audio_hidden_states <- self$audio_norm3(audio_hidden_states) *
    audio_scale_mlp$add(1) + audio_shift_mlp
    audio_hidden_states <- audio_hidden_states + self$audio_ff(norm_audio_hidden_states) *
    audio_gate_mlp

    list(hidden_states, audio_hidden_states)
}
)
