#' Z-Image Transformer Block Modules
#'
#' Fresh R port of the Z-Image DiT building blocks from the diffusers
#' reference (Apache-2.0,
#' src/diffusers/models/transformers/transformer_z_image.py). Z-Image is
#' a single-stream DiT: text and image tokens share one sequence and one
#' set of block weights. Each block uses sandwich RMSNorms (a learned
#' norm before AND after both the attention and the feed-forward) and a
#' scale/gate-only modulation — four chunks (scale_msa, gate_msa,
#' scale_mlp, gate_mlp), no shift, gates tanh-squashed, scales 1 + x.
#' The attention is plain joint self-attention, so the FLUX attention
#' module is reused with bias = FALSE and eps = 1e-5.
#'
#' @name dit_zimage_modules
NULL

# Modulation embedding width (ADALN_EMBED_DIM in the reference); the
# effective width is min(dim, 256).
.zimage_adaln_dim <- 256L

#' Z-Image feed-forward (SwiGLU with separate gate weights)
#'
#' w2(silu(w1(x)) * w3(x)) with all three linears bias-free. The hidden
#' width is int(dim / 3 * 8).
#'
#' @param dim Integer. Model width.
#' @param hidden_dim Integer. Hidden width.
#'
#' @export
zimage_feed_forward <- torch::nn_module(
                                        "zimage_feed_forward",
                                        initialize = function(dim, hidden_dim) {
    self$w1 <- torch::nn_linear(dim, hidden_dim, bias = FALSE)
    self$w2 <- torch::nn_linear(hidden_dim, dim, bias = FALSE)
    self$w3 <- torch::nn_linear(dim, hidden_dim, bias = FALSE)
},
                                        forward = function(x) {
    self$w2(torch::nnf_silu(self$w1(x)) * self$w3(x))
}
)

#' Z-Image transformer block
#'
#' Sandwich-norm residual block shared by the noise refiner, the context
#' refiner and the main trunk. With \code{modulation = TRUE} the block
#' carries an adaLN linear producing (scale_msa, gate_msa, scale_mlp,
#' gate_mlp); the context refiner uses \code{modulation = FALSE} and has
#' no adaLN weights at all.
#'
#' @param dim Integer. Model width.
#' @param n_heads Integer. Attention heads; head dim is dim / n_heads.
#' @param norm_eps Numeric. RMSNorm epsilon. Default 1e-5.
#' @param modulation Logical. Whether the block is timestep-modulated.
#'
#' @export
zimage_block <- torch::nn_module(
                                 "zimage_block",
                                 initialize = function(dim, n_heads, norm_eps = 1e-5,
        modulation = TRUE) {
    self$attention <- flux_attention(query_dim = dim, heads = n_heads,
                                     dim_head = dim %/% n_heads,
                                     eps = 1e-5, bias = FALSE)
    self$feed_forward <- zimage_feed_forward(dim, as.integer(dim / 3 * 8))

    self$attention_norm1 <- ltx23_rms_norm(dim, eps = norm_eps)
    self$ffn_norm1 <- ltx23_rms_norm(dim, eps = norm_eps)
    self$attention_norm2 <- ltx23_rms_norm(dim, eps = norm_eps)
    self$ffn_norm2 <- ltx23_rms_norm(dim, eps = norm_eps)

    self$modulation <- modulation
    if (modulation) {
        self$adaLN_modulation <- torch::nn_sequential(
                                                      torch::nn_linear(min(dim, .zimage_adaln_dim), 4L * dim,
                bias = TRUE)
        )
    }
},
                                 forward = function(x, freqs, adaln_input = NULL, chunk_size = NULL) {
    if (self$modulation) {
        mod <- self$adaLN_modulation(adaln_input)$unsqueeze(2L)
        chunks <- mod$chunk(4L, dim = 3L)
        scale_msa <- chunks[[1]]$add(1)
        gate_msa <- chunks[[2]]$tanh()
        scale_mlp <- chunks[[3]]$add(1)
        gate_mlp <- chunks[[4]]$tanh()

        attn_out <- self$attention(self$attention_norm1(x) * scale_msa,
                                   image_rotary_emb = freqs,
                                   chunk_size = chunk_size)
        x <- x + gate_msa * self$attention_norm2(attn_out)
        x + gate_mlp * self$ffn_norm2(
                                      self$feed_forward(self$ffn_norm1(x) * scale_mlp)
        )
    } else {
        attn_out <- self$attention(self$attention_norm1(x),
                                   image_rotary_emb = freqs,
                                   chunk_size = chunk_size)
        x <- x + self$attention_norm2(attn_out)
        x + self$ffn_norm2(self$feed_forward(self$ffn_norm1(x)))
    }
}
)

#' Z-Image final layer
#'
#' Parameterless LayerNorm scaled by 1 + adaLN(c) (scale only, no
#' shift), then the token-to-patch projection.
#'
#' @param hidden_size Integer. Model width.
#' @param out_channels Integer. Patch output dim
#'   (patch^2 * f_patch * latent channels).
#'
#' @export
zimage_final_layer <- torch::nn_module(
                                       "zimage_final_layer",
                                       initialize = function(hidden_size, out_channels) {
    self$norm_final <- torch::nn_layer_norm(hidden_size, eps = 1e-6,
                                            elementwise_affine = FALSE)
    self$linear <- torch::nn_linear(hidden_size, out_channels, bias = TRUE)
    self$adaLN_modulation <- torch::nn_sequential(
                                                  torch::nn_silu(),
                                                  torch::nn_linear(min(hidden_size, .zimage_adaln_dim), hidden_size,
            bias = TRUE)
    )
},
                                       forward = function(x, c) {
    scale <- self$adaLN_modulation(c)$add(1)$unsqueeze(2L)
    self$linear(self$norm_final(x) * scale)
}
)

#' Z-Image timestep embedder
#'
#' 256-dim cos-first sinusoid (computed in float32) through a
#' Linear-SiLU-Linear MLP. The model feeds t * t_scale with the
#' pipeline's t already in [0, 1].
#'
#' @param out_size Integer. Output width, min(dim, 256).
#' @param mid_size Integer. Hidden width. The full model uses 1024.
#' @param freq_size Integer. Sinusoid width. Default 256.
#'
#' @export
zimage_t_embedder <- torch::nn_module(
                                      "zimage_t_embedder",
                                      initialize = function(out_size, mid_size = 1024L, freq_size = 256L) {
    self$freq_size <- freq_size
    self$mlp <- torch::nn_sequential(
                                     torch::nn_linear(freq_size, mid_size, bias = TRUE),
                                     torch::nn_silu(),
                                     torch::nn_linear(mid_size, out_size, bias = TRUE)
    )
},
                                      forward = function(t) {
    t_freq <- ltx23_get_timestep_embedding(t, self$freq_size,
                                           flip_sin_to_cos = TRUE,
                                           downscale_freq_shift = 0)
    self$mlp(t_freq$to(dtype = self$mlp[[1]]$weight$dtype))
}
)
