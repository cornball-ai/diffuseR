#' FLUX.2 Transformer Building Blocks
#'
#' Fresh R port of the FLUX.2 MMDiT blocks from the diffusers reference
#' implementation (Apache-2.0,
#' src/diffusers/models/transformers/transformer_flux2.py). Key
#' differences from FLUX.1: modulation is computed ONCE at model level
#' by shared \code{flux2_modulation} projections and passed into the
#' blocks (block norms are parameterless), feed-forwards use SwiGLU with
#' the gate fused into \code{linear_in}, the single-stream block is a
#' ViT-22B-style parallel block with fully fused projections, and every
#' linear is bias-free. Module field names mirror the diffusers
#' state-dict keys 1:1. Reuses \code{flux_attention} (bias = FALSE),
#' \code{ltx23_rms_norm}, \code{.ltx23_sdpa}, and
#' \code{flux_apply_rotary_emb}.
#'
#' @name dit_flux2_modules
NULL

#' FLUX.2 shared modulation projection
#'
#' \code{linear(silu(temb))} producing \code{mod_param_sets} triples of
#' (shift, scale, gate). Computed once per forward at model level and
#' broadcast to every block. Reference: Flux2Modulation.
#'
#' @param dim Integer. Model dimension.
#' @param mod_param_sets Integer. Number of (shift, scale, gate) triples.
#' @param bias Logical.
#'
#' @return Module whose forward(temb) returns the modulation tensor
#'   \code{linear(silu(temb))}, holding \code{mod_param_sets} triples of
#'   (shift, scale, gate) along the last axis.
#'
#' @export
flux2_modulation <- torch::nn_module(
                                     "flux2_modulation",
                                     initialize = function(dim, mod_param_sets = 2L, bias = FALSE) {
    self$mod_param_sets <- mod_param_sets
    self$linear <- torch::nn_linear(dim, dim * 3L * mod_param_sets, bias = bias)
},
                                     forward = function(temb) {
    self$linear(torch::nnf_silu(temb))
}
)

# Split a modulation tensor into mod_param_sets triples of
# (shift, scale, gate), each [N, 1, dim]. Reference: Flux2Modulation.split.
.flux2_mod_split <- function(mod, mod_param_sets) {
    if (mod$ndim == 2L) {
        mod <- mod$unsqueeze(2L)
    }
    parts <- mod$chunk(3L * mod_param_sets, dim = -1L)
    lapply(seq_len(mod_param_sets), function(i) {
        parts[(3L * (i - 1L) + 1L):(3L * i)]
    })
}

#' FLUX.2 feed-forward (fused SwiGLU)
#'
#' \code{linear_in} projects to twice the inner dim; SwiGLU gates the
#' first half with SiLU and multiplies by the second half;
#' \code{linear_out} projects back. Reference: Flux2FeedForward +
#' Flux2SwiGLU.
#'
#' @param dim Integer. Input dimension.
#' @param dim_out Integer. Output dimension (defaults to \code{dim}).
#' @param mult Numeric. Inner dim multiplier (FLUX.2: 3.0).
#' @param bias Logical.
#'
#' @return Module whose forward(x) returns the SwiGLU-gated projection
#'   of \code{x}, a tensor with the last axis of width \code{dim_out}.
#'
#' @export
flux2_feed_forward <- torch::nn_module(
                                       "flux2_feed_forward",
                                       initialize = function(dim, dim_out = NULL, mult = 3.0, bias = FALSE) {
    inner_dim <- as.integer(dim * mult)
    self$linear_in <- torch::nn_linear(dim, inner_dim * 2L, bias = bias)
    self$linear_out <- torch::nn_linear(inner_dim, dim_out %||% dim,
                                        bias = bias)
},
                                       forward = function(x) {
    x <- self$linear_in(x)
    half <- x$shape[length(x$shape)] %/% 2L
    x <- torch::nnf_silu(x$narrow(-1L, 1L, half)) *
    x$narrow(-1L, half + 1L, half)
    self$linear_out(x)
}
)

#' FLUX.2 parallel self-attention (single-stream)
#'
#' ViT-22B-style parallel block internals: one fused projection produces
#' QKV and the SwiGLU MLP input; one fused projection consumes
#' cat(attention output, MLP output). Reference:
#' Flux2ParallelSelfAttention + Flux2ParallelSelfAttnProcessor.
#'
#' @param query_dim Integer. Model dimension.
#' @param heads Integer. Attention heads.
#' @param dim_head Integer. Per-head dimension.
#' @param mlp_ratio Numeric. MLP hidden multiplier (FLUX.2: 3.0).
#' @param eps Numeric. RMS norm epsilon.
#' @param bias Logical.
#'
#' @return Module whose forward(hidden_states, image_rotary_emb,
#'   chunk_size) returns the block output [B, S, query_dim]: attention
#'   and MLP branches computed in parallel from one fused projection,
#'   concatenated, and projected back by a second fused layer.
#'
#' @export
flux2_parallel_self_attention <- torch::nn_module(
    "flux2_parallel_self_attention",
    initialize = function(query_dim, heads, dim_head, mlp_ratio = 3.0,
                          eps = 1e-6, bias = FALSE) {
    inner_dim <- heads * dim_head
    self$heads <- heads
    self$inner_dim <- inner_dim
    self$mlp_hidden_dim <- as.integer(query_dim * mlp_ratio)

    self$to_qkv_mlp_proj <- torch::nn_linear(query_dim,
        inner_dim * 3L + self$mlp_hidden_dim * 2L, bias = bias)
    self$norm_q <- ltx23_rms_norm(dim_head, eps = eps)
    self$norm_k <- ltx23_rms_norm(dim_head, eps = eps)
    self$to_out <- torch::nn_linear(inner_dim + self$mlp_hidden_dim,
                                    query_dim, bias = bias)
},
    forward = function(hidden_states, image_rotary_emb = NULL,
                       chunk_size = NULL) {
    proj <- self$to_qkv_mlp_proj(hidden_states)
    qkv <- proj$narrow(-1L, 1L, 3L * self$inner_dim)
    mlp <- proj$narrow(-1L, 3L * self$inner_dim + 1L, self$mlp_hidden_dim * 2L)

    parts <- qkv$chunk(3L, dim = -1L)
    query <- parts[[1]]$unflatten(3L, c(self$heads, -1L))
    key <- parts[[2]]$unflatten(3L, c(self$heads, -1L))
    value <- parts[[3]]$unflatten(3L, c(self$heads, -1L))

    query <- self$norm_q(query)
    key <- self$norm_k(key)

    # [B, S, H, D] -> [B, H, S, D]
    query <- query$transpose(2L, 3L)
    key <- key$transpose(2L, 3L)
    value <- value$transpose(2L, 3L)
    if (!is.null(image_rotary_emb)) {
        query <- flux_apply_rotary_emb(query, image_rotary_emb)
        key <- flux_apply_rotary_emb(key, image_rotary_emb)
    }
    attn <- .ltx23_sdpa(query, key, value, chunk_size = chunk_size)
    attn <- attn$transpose(2L, 3L)$flatten(start_dim = 3L)
    attn <- attn$to(dtype = hidden_states$dtype)

    # SwiGLU on the fused MLP half
    half <- self$mlp_hidden_dim
    mlp <- torch::nnf_silu(mlp$narrow(-1L, 1L, half)) *
    mlp$narrow(-1L, half + 1L, half)

    # Attention half first, then the MLP half
    self$to_out(torch::torch_cat(list(attn, mlp), dim = -1L))
}
)

#' FLUX.2 double-stream (MMDiT) block
#'
#' Image and text streams with externally supplied (shift, scale, gate)
#' modulation triples, joint attention (txt first), and SwiGLU
#' feed-forwards. Reference: Flux2TransformerBlock.
#'
#' @param dim Integer. Model dimension.
#' @param num_attention_heads Integer. Attention heads.
#' @param attention_head_dim Integer. Per-head dimension.
#' @param mlp_ratio Numeric. FF multiplier (FLUX.2: 3.0).
#' @param eps Numeric. Norm epsilon.
#' @param bias Logical.
#'
#' @return Module whose forward(hidden_states, encoder_hidden_states,
#'   temb_mod_img, temb_mod_txt, image_rotary_emb) returns
#'   \code{list(encoder_hidden_states, hidden_states)}.
#'
#' @export
flux2_double_block <- torch::nn_module(
                                       "flux2_double_block",
                                       initialize = function(dim, num_attention_heads, attention_head_dim,
        mlp_ratio = 3.0, eps = 1e-6, bias = FALSE) {
    self$norm1 <- torch::nn_layer_norm(dim, eps = eps,
                                       elementwise_affine = FALSE)
    self$norm1_context <- torch::nn_layer_norm(dim, eps = eps,
        elementwise_affine = FALSE)
    self$attn <- flux_attention(dim, num_attention_heads,
                                attention_head_dim, added_kv = TRUE,
                                eps = eps, bias = bias)
    self$norm2 <- torch::nn_layer_norm(dim, eps = eps,
                                       elementwise_affine = FALSE)
    self$ff <- flux2_feed_forward(dim, dim, mult = mlp_ratio, bias = bias)
    self$norm2_context <- torch::nn_layer_norm(dim, eps = eps,
        elementwise_affine = FALSE)
    self$ff_context <- flux2_feed_forward(dim, dim, mult = mlp_ratio,
        bias = bias)
},
                                       forward = function(hidden_states, encoder_hidden_states, temb_mod_img,
        temb_mod_txt, image_rotary_emb = NULL,
        chunk_size = NULL) {
    mi <- .flux2_mod_split(temb_mod_img, 2L)
    mt <- .flux2_mod_split(temb_mod_txt, 2L)
    # Each triple is (shift, scale, gate)
    msa <- mi[[1]]
    mlp <- mi[[2]]
    c_msa <- mt[[1]]
    c_mlp <- mt[[2]]

    norm_h <- self$norm1(hidden_states) * msa[[2]]$add(1) + msa[[1]]
    norm_c <- self$norm1_context(encoder_hidden_states) *
    c_msa[[2]]$add(1) + c_msa[[1]]

    attn_out <- self$attn(
                          hidden_states = norm_h,
                          encoder_hidden_states = norm_c,
                          image_rotary_emb = image_rotary_emb,
                          chunk_size = chunk_size
    )

    hidden_states <- hidden_states + msa[[3]] * attn_out[[1]]
    norm_h <- self$norm2(hidden_states) * mlp[[2]]$add(1) + mlp[[1]]
    hidden_states <- hidden_states + mlp[[3]] * self$ff(norm_h)

    encoder_hidden_states <- encoder_hidden_states + c_msa[[3]] * attn_out[[2]]
    norm_c <- self$norm2_context(encoder_hidden_states) *
    c_mlp[[2]]$add(1) + c_mlp[[1]]
    encoder_hidden_states <- encoder_hidden_states +
    c_mlp[[3]] * self$ff_context(norm_c)

    if (encoder_hidden_states$dtype == torch::torch_float16()) {
        encoder_hidden_states <- encoder_hidden_states$clamp(-65504, 65504)
    }
    list(encoder_hidden_states, hidden_states)
}
)

#' FLUX.2 single-stream block (parallel)
#'
#' Parameterless LayerNorm with external modulation, then the fused
#' parallel attention+MLP. Operates on the pre-concatenated [text; image]
#' sequence (the reference model concatenates once before the stack).
#' Reference: Flux2SingleTransformerBlock.
#'
#' @param dim Integer. Model dimension.
#' @param num_attention_heads Integer. Attention heads.
#' @param attention_head_dim Integer. Per-head dimension.
#' @param mlp_ratio Numeric. MLP multiplier (FLUX.2: 3.0).
#' @param eps Numeric. Norm epsilon.
#' @param bias Logical.
#'
#' @return Module whose forward(hidden_states, temb_mod,
#'   image_rotary_emb) returns the joint hidden states.
#'
#' @export
flux2_single_block <- torch::nn_module(
                                       "flux2_single_block",
                                       initialize = function(dim, num_attention_heads, attention_head_dim,
        mlp_ratio = 3.0, eps = 1e-6, bias = FALSE) {
    self$norm <- torch::nn_layer_norm(dim, eps = eps,
                                      elementwise_affine = FALSE)
    self$attn <- flux2_parallel_self_attention(
        dim, num_attention_heads, attention_head_dim,
        mlp_ratio = mlp_ratio, eps = eps, bias = bias
    )
},
                                       forward = function(hidden_states, temb_mod, image_rotary_emb = NULL,
        chunk_size = NULL) {
    mod <- .flux2_mod_split(temb_mod, 1L)[[1]]
    norm_h <- self$norm(hidden_states) * mod[[2]]$add(1) + mod[[1]]
    attn_out <- self$attn(
                          hidden_states = norm_h,
                          image_rotary_emb = image_rotary_emb,
                          chunk_size = chunk_size
    )
    hidden_states <- hidden_states + mod[[3]] * attn_out
    if (hidden_states$dtype == torch::torch_float16()) {
        hidden_states <- hidden_states$clamp(-65504, 65504)
    }
    hidden_states
}
)
