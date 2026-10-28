#' FLUX 3 Transformer Building Blocks
#'
#' Fresh R port of the FLUX 3 blocks from the diffusers reference
#' implementation (Apache-2.0,
#' src/diffusers/models/transformers/transformer_flux3.py). Every block
#' is a parallel attention + SwiGLU feed-forward computed from one
#' modulated input: per-mode blocks self-attend within one token mode
#' (image, reference image, or text), joint blocks attend over the
#' concatenated [text; content] sequence. Modulation is per token,
#' computed at model level. Module field names mirror the diffusers
#' state-dict keys 1:1; every linear is bias-free. Reuses
#' \code{flux2_feed_forward}, \code{flux_apply_rotary_emb} and
#' \code{.ltx23_sdpa}.
#'
#' @name dit_flux3_modules
NULL

#' FLUX 3 self-attention
#'
#' Separate q/k/v projections, per-head RMS norms and rotary embeddings
#' applied in float32 (cast back to the value dtype before attention),
#' and one output projection. Reference: Flux3Attention +
#' Flux3AttnProcessor.
#'
#' @param query_dim Integer. Model dimension.
#' @param heads Integer. Attention heads.
#' @param dim_head Integer. Per-head dimension.
#' @param eps Numeric. RMS norm epsilon.
#'
#' @return Module whose forward(hidden_states, image_rotary_emb,
#'   chunk_size) returns the attended states [B, S, query_dim].
#'
#' @export
flux3_attention <- torch::nn_module(
                                    "flux3_attention",
                                    initialize = function(query_dim, heads, dim_head, eps = 1e-6) {
    inner_dim <- heads * dim_head
    self$heads <- heads
    self$eps <- eps
    self$to_q <- linear_noinit(query_dim, inner_dim, bias = FALSE)
    self$to_k <- linear_noinit(query_dim, inner_dim, bias = FALSE)
    self$to_v <- linear_noinit(query_dim, inner_dim, bias = FALSE)
    self$norm_q <- ltx23_rms_norm(dim_head, eps = eps)
    self$norm_k <- ltx23_rms_norm(dim_head, eps = eps)
    self$to_out <- torch::nn_module_list(list(
            linear_noinit(inner_dim, query_dim, bias = FALSE)))
},
                                    forward = function(hidden_states, image_rotary_emb = NULL,
        chunk_size = NULL) {
    value <- self$to_v(hidden_states)$unflatten(3L, c(self$heads, -1L))
    query <- self$to_q(hidden_states)$unflatten(3L, c(self$heads, -1L))
    key <- self$to_k(hidden_states)$unflatten(3L, c(self$heads, -1L))

    # [B, S, H, D] -> [B, H, S, D]; norms and rotary in float32
    query <- .flux3_rms_norm_f32(query, self$norm_q)$transpose(2L, 3L)
    key <- .flux3_rms_norm_f32(key, self$norm_k)$transpose(2L, 3L)
    value <- value$transpose(2L, 3L)
    if (!is.null(image_rotary_emb)) {
        query <- flux_apply_rotary_emb(query, image_rotary_emb)
        key <- flux_apply_rotary_emb(key, image_rotary_emb)
    }
    query <- query$to(dtype = value$dtype)
    key <- key$to(dtype = value$dtype)

    attn <- .ltx23_sdpa(query, key, value, chunk_size = chunk_size)
    attn <- attn$transpose(2L, 3L)$flatten(start_dim = 3L)
    self$to_out[[1]](attn$to(dtype = hidden_states$dtype))
}
)

# torch.nn.RMSNorm applied to a float32 copy of x: the reference calls
# norm_q(query.float()), so the weight multiplies in float32 too.
.flux3_rms_norm_f32 <- function(x, norm) {
    x <- x$to(dtype = torch::torch_float32())
    variance <- x$pow(2)$mean(dim = -1L, keepdim = TRUE)
    x * torch::torch_rsqrt(variance + norm$eps) *
    norm$weight$to(dtype = torch::torch_float32())
}

# Split a per-token modulation tensor [B, S or 1, 3 * dim] into
# (shift, scale, gate). Reference: torch.chunk(temb_mod, 3, dim=-1).
.flux3_mod_split <- function(mod) {
    mod$chunk(3L, dim = -1L)
}

#' FLUX 3 per-mode transformer block
#'
#' Parameterless LayerNorm, per-token (shift, scale) modulation, then
#' parallel attention and feed-forward from the same normed input, added
#' back under one gate. Self-attention stays within a single token mode.
#' Reference: Flux3ModeTransformerBlock.
#'
#' @param dim Integer. Model dimension.
#' @param num_attention_heads Integer. Attention heads.
#' @param attention_head_dim Integer. Per-head dimension.
#' @param mlp_ratio Numeric. Feed-forward multiplier (3.0).
#' @param eps Numeric. Norm epsilon.
#'
#' @return Module whose forward(hidden_states, temb_mod,
#'   image_rotary_emb, chunk_size) returns the updated states.
#'
#' @export
flux3_mode_block <- torch::nn_module(
                                     "flux3_mode_block",
                                     initialize = function(dim, num_attention_heads, attention_head_dim,
        mlp_ratio = 3.0, eps = 1e-6) {
    self$norm <- torch::nn_layer_norm(dim, eps = eps,
                                      elementwise_affine = FALSE)
    self$attn <- flux3_attention(dim, num_attention_heads,
                                 attention_head_dim, eps = eps)
    self$ff <- flux2_feed_forward(dim, dim, mult = mlp_ratio)
},
                                     forward = function(hidden_states, temb_mod, image_rotary_emb = NULL,
        chunk_size = NULL) {
    mod <- .flux3_mod_split(temb_mod)
    norm_h <- self$norm(hidden_states) * mod[[2]]$add(1) + mod[[1]]
    attn_out <- self$attn(norm_h, image_rotary_emb = image_rotary_emb,
                          chunk_size = chunk_size)
    hidden_states + mod[[3]] * (attn_out + self$ff(norm_h))
}
)

#' FLUX 3 joint (single-stream) block
#'
#' Text and content tokens are normed and modulated separately, then
#' concatenated (text first) for one parallel attention + feed-forward;
#' the output is split back and added under per-stream gates.
#' Reference: Flux3SingleTransformerBlock.
#'
#' @inheritParams flux3_mode_block
#'
#' @return Module whose forward(hidden_states, encoder_hidden_states,
#'   temb_mod, temb_mod_context, image_rotary_emb, chunk_size) returns
#'   \code{list(encoder_hidden_states, hidden_states)}.
#'
#' @export
flux3_single_block <- torch::nn_module(
                                       "flux3_single_block",
                                       initialize = function(dim, num_attention_heads, attention_head_dim,
        mlp_ratio = 3.0, eps = 1e-6) {
    self$norm <- torch::nn_layer_norm(dim, eps = eps,
                                      elementwise_affine = FALSE)
    self$attn <- flux3_attention(dim, num_attention_heads,
                                 attention_head_dim, eps = eps)
    self$ff <- flux2_feed_forward(dim, dim, mult = mlp_ratio)
},
                                       forward = function(hidden_states, encoder_hidden_states, temb_mod,
        temb_mod_context, image_rotary_emb = NULL,
        chunk_size = NULL) {
    mod <- .flux3_mod_split(temb_mod)
    c_mod <- .flux3_mod_split(temb_mod_context)

    norm_h <- self$norm(hidden_states) * mod[[2]]$add(1) + mod[[1]]
    norm_c <- self$norm(encoder_hidden_states) * c_mod[[2]]$add(1) +
    c_mod[[1]]
    combined <- torch::torch_cat(list(norm_c, norm_h), dim = 2L)

    output <- self$attn(combined, image_rotary_emb = image_rotary_emb,
                        chunk_size = chunk_size) + self$ff(combined)
    txt_len <- encoder_hidden_states$shape[2]
    context_out <- output$narrow(2L, 1L, txt_len)
    content_out <- output$narrow(2L, txt_len + 1L, hidden_states$shape[2])

    list(encoder_hidden_states + c_mod[[3]] * context_out,
         hidden_states + mod[[3]] * content_out)
}
)
