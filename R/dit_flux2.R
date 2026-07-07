#' FLUX.2 Transformer (MMDiT)
#'
#' Fresh R port of Flux2Transformer2DModel from the diffusers reference
#' implementation (Apache-2.0,
#' src/diffusers/models/transformers/transformer_flux2.py). Defaults are
#' the klein-4B configuration (5 double + 20 single blocks). Guidance
#' embeddings (FLUX.2-dev) are not implemented; klein is step-distilled
#' with \code{guidance_embeds = false}. Timestep conditioning has no
#' pooled-text component, and the three modulation projections are
#' shared across all blocks.
#'
#' @name dit_flux2
NULL

# Timestep-only conditioning: sinusoid(256) -> MLP, bias-free. Matches
# diffusers Flux2TimestepGuidanceEmbeddings state-dict names (klein has
# no guidance_embedder).
flux2_time_guidance_embed <- torch::nn_module(
    "flux2_time_guidance_embed",
    initialize = function(embedding_dim, in_channels = 256L) {
    self$in_channels <- in_channels
    self$timestep_embedder <- ltx23_timestep_embedding(in_channels,
                                                       embedding_dim,
                                                       bias = FALSE)
},
    forward = function(timestep) {
    proj <- ltx23_get_timestep_embedding(timestep, self$in_channels,
        flip_sin_to_cos = TRUE, downscale_freq_shift = 0)
    self$timestep_embedder(proj$to(dtype = self$timestep_embedder$linear_1$weight$dtype))
}
)

#' FLUX.2 transformer model
#'
#' Shared modulation computed once per forward; double blocks over
#' separate text/image streams, then single (parallel) blocks over the
#' concatenated [text; image] sequence. Rotary embeddings are
#' precomputed by the caller with \code{\link{flux_pos_embed}}
#' (\code{axes_dim = c(32, 32, 32, 32)}, \code{theta = 2000}) over the
#' concatenated [text; image] 4-axis position ids.
#'
#' @param in_channels Integer. Packed latent channels (128).
#' @param num_layers Integer. Double-stream block count (klein-4B: 5).
#' @param num_single_layers Integer. Single-stream block count (20).
#' @param attention_head_dim Integer. Per-head dimension.
#' @param num_attention_heads Integer. Attention heads.
#' @param joint_attention_dim Integer. Text embedding dim (7680).
#' @param mlp_ratio Numeric. Feed-forward multiplier (3.0).
#' @param timestep_guidance_channels Integer. Sinusoid width (256).
#' @param axes_dims_rope Integer vector. Per-axis rotary dims.
#' @param rope_theta Numeric. Rotary base frequency (2000).
#' @param eps Numeric. Norm epsilon.
#' @param out_channels Integer or NULL. Defaults to \code{in_channels}.
#'
#' @return Module whose forward(hidden_states, encoder_hidden_states,
#'   timestep, image_rotary_emb) returns the predicted velocity for the
#'   image tokens [B, S_img, out_channels]. \code{timestep} is in sigma
#'   space (0-1); it is scaled by 1000 internally.
#'
#' @export
flux2_transformer <- torch::nn_module(
    "flux2_transformer",
    initialize = function(in_channels = 128L,
                          num_layers = 5L,
                          num_single_layers = 20L,
                          attention_head_dim = 128L,
                          num_attention_heads = 24L,
                          joint_attention_dim = 7680L,
                          mlp_ratio = 3.0,
                          timestep_guidance_channels = 256L,
                          axes_dims_rope = c(32L, 32L, 32L, 32L),
                          rope_theta = 2000,
                          eps = 1e-6,
                          out_channels = NULL) {
    inner_dim <- num_attention_heads * attention_head_dim
    self$inner_dim <- inner_dim
    self$axes_dims_rope <- as.integer(axes_dims_rope)
    self$rope_theta <- rope_theta
    self$out_channels <- as.integer(out_channels %||% in_channels)

    self$time_guidance_embed <- flux2_time_guidance_embed(
                                                          inner_dim, as.integer(timestep_guidance_channels)
    )
    self$double_stream_modulation_img <- flux2_modulation(inner_dim, 2L)
    self$double_stream_modulation_txt <- flux2_modulation(inner_dim, 2L)
    self$single_stream_modulation <- flux2_modulation(inner_dim, 1L)

    self$x_embedder <- torch::nn_linear(in_channels, inner_dim,
                                        bias = FALSE)
    self$context_embedder <- torch::nn_linear(joint_attention_dim,
                                              inner_dim, bias = FALSE)

    self$transformer_blocks <- torch::nn_module_list(
                                                     lapply(seq_len(num_layers), function(i) {
        flux2_double_block(inner_dim, num_attention_heads,
                           attention_head_dim, mlp_ratio = mlp_ratio,
                           eps = eps)
    })
    )
    self$single_transformer_blocks <- torch::nn_module_list(
                                                            lapply(seq_len(num_single_layers), function(i) {
        flux2_single_block(inner_dim, num_attention_heads,
                           attention_head_dim, mlp_ratio = mlp_ratio,
                           eps = eps)
    })
    )

    self$norm_out <- flux_ada_layer_norm_continuous(inner_dim, inner_dim,
                                                    bias = FALSE)
    self$proj_out <- torch::nn_linear(inner_dim, self$out_channels,
                                      bias = FALSE)
},
    forward = function(hidden_states, encoder_hidden_states, timestep,
                       image_rotary_emb, chunk_size = NULL) {
    hidden_states <- self$x_embedder(hidden_states)
    timestep <- timestep$to(dtype = hidden_states$dtype)$mul(1000)
    temb <- self$time_guidance_embed(timestep)

    mod_img <- self$double_stream_modulation_img(temb)
    mod_txt <- self$double_stream_modulation_txt(temb)
    mod_single <- self$single_stream_modulation(temb)

    encoder_hidden_states <- self$context_embedder(encoder_hidden_states)

    block_gc <- isTRUE(getOption("diffuseR.block_gc"))
    for (i in seq_along(self$transformer_blocks)) {
        res <- self$transformer_blocks[[i]](
                                            hidden_states = hidden_states,
                                            encoder_hidden_states = encoder_hidden_states,
                                            temb_mod_img = mod_img,
                                            temb_mod_txt = mod_txt,
                                            image_rotary_emb = image_rotary_emb,
                                            chunk_size = chunk_size
        )
        encoder_hidden_states <- res[[1]]
        hidden_states <- res[[2]]
        if (block_gc) {
            gc(verbose = FALSE)
        }
    }

    txt_len <- encoder_hidden_states$shape[2]
    hidden_states <- torch::torch_cat(
                                      list(encoder_hidden_states, hidden_states),
                                      dim = 2L
    )
    for (i in seq_along(self$single_transformer_blocks)) {
        hidden_states <- self$single_transformer_blocks[[i]](
                                                             hidden_states = hidden_states,
                                                             temb_mod = mod_single,
                                                             image_rotary_emb = image_rotary_emb,
                                                             chunk_size = chunk_size
        )
        if (block_gc) {
            gc(verbose = FALSE)
        }
    }
    hidden_states <- hidden_states$narrow(
                                          2L, txt_len + 1L,
                                          hidden_states$shape[2] - txt_len
    )

    hidden_states <- self$norm_out(hidden_states, temb)
    self$proj_out(hidden_states)
}
)
