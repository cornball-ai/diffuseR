#' FLUX Transformer Building Blocks
#'
#' Fresh R port of the FLUX MMDiT blocks from the diffusers reference
#' implementation (Apache-2.0,
#' src/diffusers/models/transformers/transformer_flux.py and
#' src/diffusers/models/normalization.py). Module field names mirror the
#' diffusers state-dict keys 1:1 so checkpoints load without remapping.
#' Reuses the LTX primitives \code{ltx23_rms_norm}, \code{.ltx23_sdpa}
#' and \code{ltx23_feed_forward}.
#'
#' @name dit_flux_modules
NULL

#' FLUX adaLN-Zero modulation (double-stream)
#'
#' Projects the conditioning embedding to six modulation vectors and
#' returns the msa-modulated input plus the remaining parameters.
#' Reference: diffusers AdaLayerNormZero.
#'
#' @param dim Integer. Model dimension.
#'
#' @return Module whose forward(x, emb) returns
#'   \code{list(x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp)}.
#'
#' @export
flux_ada_layer_norm_zero <- torch::nn_module(
    "flux_ada_layer_norm_zero",
    initialize = function(dim) {
    self$linear <- torch::nn_linear(dim, 6L * dim, bias = TRUE)
    self$norm <- torch::nn_layer_norm(dim, eps = 1e-6,
                                      elementwise_affine = FALSE)
},
    forward = function(x, emb) {
    emb <- self$linear(torch::nnf_silu(emb))
    p <- emb$chunk(6L, dim = 2L)
    # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
    x <- self$norm(x) * p[[2]]$unsqueeze(2L)$add(1) + p[[1]]$unsqueeze(2L)
    list(x, p[[3]], p[[4]], p[[5]], p[[6]])
}
)

#' FLUX adaLN-Zero modulation (single-stream)
#'
#' Three modulation vectors: shift, scale, gate. Reference: diffusers
#' AdaLayerNormZeroSingle.
#'
#' @param dim Integer. Model dimension.
#'
#' @return Module whose forward(x, emb) returns \code{list(x_norm, gate)}.
#'
#' @export
flux_ada_layer_norm_zero_single <- torch::nn_module(
    "flux_ada_layer_norm_zero_single",
    initialize = function(dim) {
    self$linear <- torch::nn_linear(dim, 3L * dim, bias = TRUE)
    self$norm <- torch::nn_layer_norm(dim, eps = 1e-6,
                                      elementwise_affine = FALSE)
},
    forward = function(x, emb) {
    emb <- self$linear(torch::nnf_silu(emb))
    p <- emb$chunk(3L, dim = 2L)
    # shift_msa, scale_msa, gate_msa
    x <- self$norm(x) * p[[2]]$unsqueeze(2L)$add(1) + p[[1]]$unsqueeze(2L)
    list(x, p[[3]])
}
)

#' FLUX continuous adaLN (final norm)
#'
#' Scale/shift conditioning of the final norm. Note the chunk order:
#' scale first, then shift (the reverse of adaLN-Zero). Reference:
#' diffusers AdaLayerNormContinuous as used by FLUX norm_out
#' (elementwise_affine = FALSE, eps = 1e-6).
#'
#' @param dim Integer. Model dimension.
#' @param cond_dim Integer. Conditioning embedding dimension.
#' @param bias Logical. Bias on the projection (TRUE for FLUX.1, FALSE
#'   for FLUX.2).
#'
#' @export
flux_ada_layer_norm_continuous <- torch::nn_module(
    "flux_ada_layer_norm_continuous",
    initialize = function(dim, cond_dim = dim, bias = TRUE) {
    self$linear <- torch::nn_linear(cond_dim, 2L * dim, bias = bias)
    self$norm <- torch::nn_layer_norm(dim, eps = 1e-6,
                                      elementwise_affine = FALSE)
},
    forward = function(x, cond) {
    emb <- self$linear(torch::nnf_silu(cond)$to(dtype = x$dtype))
    p <- emb$chunk(2L, dim = 2L)
    # scale, shift
    self$norm(x) * p[[1]]$unsqueeze(2L)$add(1) + p[[2]]$unsqueeze(2L)
}
)

#' FLUX joint attention
#'
#' Multi-head attention with per-head RMS q/k norms and rotary position
#' embeddings. With \code{added_kv = TRUE} (double-stream blocks) the
#' text stream gets its own q/k/v projections and both streams attend
#' jointly (text tokens first); the outputs are split back and projected
#' per stream. With \code{pre_only = TRUE} (single-stream blocks) there
#' is no output projection. Reference: diffusers FluxAttention +
#' FluxAttnProcessor.
#'
#' @param query_dim Integer. Model dimension.
#' @param heads Integer. Number of attention heads.
#' @param dim_head Integer. Per-head dimension.
#' @param added_kv Logical. Add text-stream projections (double blocks).
#' @param pre_only Logical. Skip the output projection (single blocks).
#' @param eps Numeric. RMS norm epsilon.
#' @param bias Logical. Bias on the linear projections (TRUE for FLUX.1,
#'   FALSE for FLUX.2).
#'
#' @export
flux_attention <- torch::nn_module(
                                   "flux_attention",
                                   initialize = function(query_dim, heads, dim_head, added_kv = FALSE,
        pre_only = FALSE, eps = 1e-6, bias = TRUE) {
    inner_dim <- heads * dim_head
    self$heads <- heads
    self$dim_head <- dim_head
    self$added_kv <- added_kv
    self$pre_only <- pre_only

    self$norm_q <- ltx23_rms_norm(dim_head, eps = eps)
    self$norm_k <- ltx23_rms_norm(dim_head, eps = eps)
    self$to_q <- torch::nn_linear(query_dim, inner_dim, bias = bias)
    self$to_k <- torch::nn_linear(query_dim, inner_dim, bias = bias)
    self$to_v <- torch::nn_linear(query_dim, inner_dim, bias = bias)

    if (!pre_only) {
        self$to_out <- torch::nn_module_list(list(
                torch::nn_linear(inner_dim, query_dim, bias = bias)
            ))
    }
    if (added_kv) {
        self$norm_added_q <- ltx23_rms_norm(dim_head, eps = eps)
        self$norm_added_k <- ltx23_rms_norm(dim_head, eps = eps)
        self$add_q_proj <- torch::nn_linear(query_dim, inner_dim, bias = bias)
        self$add_k_proj <- torch::nn_linear(query_dim, inner_dim, bias = bias)
        self$add_v_proj <- torch::nn_linear(query_dim, inner_dim, bias = bias)
        self$to_add_out <- torch::nn_linear(inner_dim, query_dim, bias = bias)
    }
},
                                   forward = function(hidden_states, encoder_hidden_states = NULL,
        image_rotary_emb = NULL, chunk_size = NULL) {
    # Per-head layout [B, S, H, D]
    query <- self$to_q(hidden_states)$unflatten(3L, c(self$heads, -1L))
    key <- self$to_k(hidden_states)$unflatten(3L, c(self$heads, -1L))
    value <- self$to_v(hidden_states)$unflatten(3L, c(self$heads, -1L))

    query <- self$norm_q(query)
    key <- self$norm_k(key)

    if (!is.null(encoder_hidden_states)) {
        txt_len <- encoder_hidden_states$shape[2]
        eq <- self$add_q_proj(encoder_hidden_states)$unflatten(3L,
            c(self$heads, -1L))
        ek <- self$add_k_proj(encoder_hidden_states)$unflatten(3L, c(self$heads, -1L))
        ev <- self$add_v_proj(encoder_hidden_states)$unflatten(3L, c(self$heads, -1L))
        eq <- self$norm_added_q(eq)
        ek <- self$norm_added_k(ek)
        # Text tokens first, matching the rotary frequency layout
        query <- torch::torch_cat(list(eq, query), dim = 2L)
        key <- torch::torch_cat(list(ek, key), dim = 2L)
        value <- torch::torch_cat(list(ev, value), dim = 2L)
    }

    # [B, S, H, D] -> [B, H, S, D] for RoPE and attention
    query <- query$transpose(2L, 3L)
    key <- key$transpose(2L, 3L)
    value <- value$transpose(2L, 3L)

    if (!is.null(image_rotary_emb)) {
        query <- flux_apply_rotary_emb(query, image_rotary_emb)
        key <- flux_apply_rotary_emb(key, image_rotary_emb)
    }

    out <- .ltx23_sdpa(query, key, value, chunk_size = chunk_size)
    # [B, H, S, D] -> [B, S, H*D]
    out <- out$transpose(2L, 3L)$flatten(start_dim = 3L)
    out <- out$to(dtype = hidden_states$dtype)

    if (!is.null(encoder_hidden_states)) {
        seq_len <- out$shape[2]
        ctx <- out$narrow(2L, 1L, txt_len)
        img <- out$narrow(2L, txt_len + 1L, seq_len - txt_len)
        return(list(
                    self$to_out[[1]](img$contiguous()),
                    self$to_add_out(ctx$contiguous())
            ))
    }
    if (self$pre_only) {
        return(out)
    }
    self$to_out[[1]](out)
}
)

#' FLUX double-stream (MMDiT) transformer block
#'
#' Image and text streams each get adaLN-Zero modulation and a
#' feed-forward; attention is joint across both streams. Reference:
#' diffusers FluxTransformerBlock.
#'
#' @param dim Integer. Model dimension.
#' @param num_attention_heads Integer. Attention heads.
#' @param attention_head_dim Integer. Per-head dimension.
#'
#' @return Module whose forward(hidden_states, encoder_hidden_states,
#'   temb, image_rotary_emb) returns
#'   \code{list(encoder_hidden_states, hidden_states)}.
#'
#' @export
flux_double_block <- torch::nn_module(
                                      "flux_double_block",
                                      initialize = function(dim, num_attention_heads, attention_head_dim) {
    self$norm1 <- flux_ada_layer_norm_zero(dim)
    self$norm1_context <- flux_ada_layer_norm_zero(dim)
    self$attn <- flux_attention(dim, num_attention_heads, attention_head_dim,
                                added_kv = TRUE)
    self$norm2 <- torch::nn_layer_norm(dim, eps = 1e-6,
                                       elementwise_affine = FALSE)
    self$ff <- ltx23_feed_forward(dim, mult = 4L)
    self$norm2_context <- torch::nn_layer_norm(dim, eps = 1e-6,
        elementwise_affine = FALSE)
    self$ff_context <- ltx23_feed_forward(dim, mult = 4L)
},
                                      forward = function(hidden_states, encoder_hidden_states, temb,
        image_rotary_emb = NULL, chunk_size = NULL) {
    n1 <- self$norm1(hidden_states, emb = temb)
    n1c <- self$norm1_context(encoder_hidden_states, emb = temb)

    attn_out <- self$attn(
                          hidden_states = n1[[1]],
                          encoder_hidden_states = n1c[[1]],
                          image_rotary_emb = image_rotary_emb,
                          chunk_size = chunk_size
    )

    # Image stream: gated attention + modulated feed-forward
    hidden_states <- hidden_states + n1[[2]]$unsqueeze(2L) * attn_out[[1]]
    norm_h <- self$norm2(hidden_states) * n1[[4]]$unsqueeze(2L)$add(1) +
    n1[[3]]$unsqueeze(2L)
    hidden_states <- hidden_states + n1[[5]]$unsqueeze(2L) * self$ff(norm_h)

    # Text stream mirrors with its own modulation
    encoder_hidden_states <- encoder_hidden_states +
    n1c[[2]]$unsqueeze(2L) * attn_out[[2]]
    norm_c <- self$norm2_context(encoder_hidden_states) *
    n1c[[4]]$unsqueeze(2L)$add(1) + n1c[[3]]$unsqueeze(2L)
    encoder_hidden_states <- encoder_hidden_states +
    n1c[[5]]$unsqueeze(2L) * self$ff_context(norm_c)

    if (encoder_hidden_states$dtype == torch::torch_float16()) {
        encoder_hidden_states <- encoder_hidden_states$clamp(-65504, 65504)
    }
    list(encoder_hidden_states, hidden_states)
}
)

#' FLUX single-stream transformer block
#'
#' Parallel attention + MLP over the joint [text; image] sequence with a
#' shared gate: \code{x + gate * proj_out(cat(attn, gelu(mlp)))}. The
#' reference concatenates the streams inside every block and splits after;
#' here the caller concatenates once before the single-block stack, which
#' is numerically identical. Reference: diffusers
#' FluxSingleTransformerBlock.
#'
#' @param dim Integer. Model dimension.
#' @param num_attention_heads Integer. Attention heads.
#' @param attention_head_dim Integer. Per-head dimension.
#' @param mlp_ratio Numeric. MLP hidden dim multiplier.
#'
#' @return Module whose forward(hidden_states, temb, image_rotary_emb)
#'   returns the joint hidden states.
#'
#' @export
flux_single_block <- torch::nn_module(
                                      "flux_single_block",
                                      initialize = function(dim, num_attention_heads, attention_head_dim,
        mlp_ratio = 4.0) {
    mlp_hidden_dim <- as.integer(dim * mlp_ratio)
    self$norm <- flux_ada_layer_norm_zero_single(dim)
    self$proj_mlp <- torch::nn_linear(dim, mlp_hidden_dim)
    self$proj_out <- torch::nn_linear(dim + mlp_hidden_dim, dim)
    self$attn <- flux_attention(dim, num_attention_heads, attention_head_dim,
                                pre_only = TRUE)
},
                                      forward = function(hidden_states, temb, image_rotary_emb = NULL,
        chunk_size = NULL) {
    residual <- hidden_states
    n <- self$norm(hidden_states, emb = temb)
    mlp <- torch::nnf_gelu(self$proj_mlp(n[[1]]), approximate = "tanh")
    attn_out <- self$attn(
                          hidden_states = n[[1]],
                          image_rotary_emb = image_rotary_emb,
                          chunk_size = chunk_size
    )

    # Attention half first, then the MLP half
    hidden_states <- torch::torch_cat(list(attn_out, mlp), dim = 3L)
    hidden_states <- residual + n[[2]]$unsqueeze(2L) * self$proj_out(hidden_states)
    if (hidden_states$dtype == torch::torch_float16()) {
        hidden_states <- hidden_states$clamp(-65504, 65504)
    }
    hidden_states
}
)
