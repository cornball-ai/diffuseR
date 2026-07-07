#' FLUX Transformer (MMDiT)
#'
#' Fresh R port of FluxTransformer2DModel from the diffusers reference
#' implementation (Apache-2.0,
#' src/diffusers/models/transformers/transformer_flux.py). The module
#' tree mirrors the diffusers state-dict keys 1:1, so checkpoints load
#' without remapping. FLUX.1-schnell has no guidance embedder
#' (guidance_embeds = FALSE); the guidance-distilled dev variant is not
#' implemented.
#'
#' @name dit_flux
NULL

# Combined timestep + pooled-text conditioning, matching diffusers
# CombinedTimestepTextProjEmbeddings state-dict names. Both embedders are
# linear_1 -> silu -> linear_2, which ltx23_timestep_embedding provides
# (PixArtAlphaTextProjection with act_fn = "silu" is the same function).
flux_time_text_embed <- torch::nn_module(
    "flux_time_text_embed",
    initialize = function(embedding_dim, pooled_projection_dim) {
    self$timestep_embedder <- ltx23_timestep_embedding(256L, embedding_dim)
    self$text_embedder <- ltx23_timestep_embedding(pooled_projection_dim,
                                                   embedding_dim)
},
    forward = function(timestep, pooled_projection) {
    proj <- ltx23_get_timestep_embedding(timestep, 256L,
        flip_sin_to_cos = TRUE, downscale_freq_shift = 0)
    temb <- self$timestep_embedder(proj$to(dtype = pooled_projection$dtype))
    temb + self$text_embedder(pooled_projection)
}
)

#' FLUX transformer model
#'
#' 19 double-stream (MMDiT) blocks followed by 38 single-stream blocks
#' over the joint [text; image] sequence, with adaLN-Zero conditioning on
#' timestep + pooled CLIP text. Rotary embeddings are precomputed by the
#' caller with \code{flux_pos_embed} (they are static across denoise
#' steps). Defaults are the FLUX.1-schnell configuration.
#'
#' @param in_channels Integer. Packed latent channels (64).
#' @param num_layers Integer. Double-stream block count.
#' @param num_single_layers Integer. Single-stream block count.
#' @param attention_head_dim Integer. Per-head dimension.
#' @param num_attention_heads Integer. Attention heads.
#' @param joint_attention_dim Integer. T5 embedding dim (4096).
#' @param pooled_projection_dim Integer. CLIP pooled dim (768).
#' @param axes_dims_rope Integer vector. Per-axis rotary dims.
#' @param out_channels Integer or NULL. Output channels (defaults to
#'   \code{in_channels}).
#'
#' @return Module whose forward(hidden_states, encoder_hidden_states,
#'   pooled_projections, timestep, image_rotary_emb) returns the
#'   predicted velocity for the image tokens [B, S_img, out_channels].
#'   \code{timestep} is in sigma space (0-1); it is scaled by 1000
#'   internally, matching the reference.
#'
#' @export
flux_transformer <- torch::nn_module(
    "flux_transformer",
    initialize = function(in_channels = 64L,
                          num_layers = 19L,
                          num_single_layers = 38L,
                          attention_head_dim = 128L,
                          num_attention_heads = 24L,
                          joint_attention_dim = 4096L,
                          pooled_projection_dim = 768L,
                          axes_dims_rope = c(16L, 56L, 56L),
                          out_channels = NULL) {
    inner_dim <- num_attention_heads * attention_head_dim
    self$inner_dim <- inner_dim
    self$axes_dims_rope <- as.integer(axes_dims_rope)
    self$out_channels <- as.integer(out_channels %||% in_channels)

    self$time_text_embed <- flux_time_text_embed(inner_dim,
                                                 pooled_projection_dim)
    self$context_embedder <- torch::nn_linear(joint_attention_dim, inner_dim)
    self$x_embedder <- torch::nn_linear(in_channels, inner_dim)

    self$transformer_blocks <- torch::nn_module_list(
                                                     lapply(seq_len(num_layers), function(i) {
        flux_double_block(inner_dim, num_attention_heads,
                          attention_head_dim)
    })
    )
    self$single_transformer_blocks <- torch::nn_module_list(
                                                            lapply(seq_len(num_single_layers), function(i) {
        flux_single_block(inner_dim, num_attention_heads,
                          attention_head_dim)
    })
    )

    self$norm_out <- flux_ada_layer_norm_continuous(inner_dim, inner_dim)
    self$proj_out <- torch::nn_linear(inner_dim, self$out_channels,
                                      bias = TRUE)
},
    forward = function(hidden_states, encoder_hidden_states,
                       pooled_projections, timestep, image_rotary_emb,
                       chunk_size = NULL) {
    hidden_states <- self$x_embedder(hidden_states)
    timestep <- timestep$to(dtype = hidden_states$dtype)$mul(1000)
    temb <- self$time_text_embed(timestep, pooled_projections)
    encoder_hidden_states <- self$context_embedder(encoder_hidden_states)

    block_gc <- isTRUE(getOption("diffuseR.block_gc"))
    debug <- isTRUE(getOption("diffuseR.debug"))

    for (i in seq_along(self$transformer_blocks)) {
        res <- self$transformer_blocks[[i]](
                                            hidden_states = hidden_states,
                                            encoder_hidden_states = encoder_hidden_states,
                                            temb = temb,
                                            image_rotary_emb = image_rotary_emb,
                                            chunk_size = chunk_size
        )
        encoder_hidden_states <- res[[1]]
        hidden_states <- res[[2]]
        if (debug && torch::cuda_is_available()) {
            ms <- torch::cuda_memory_stats()
            message(sprintf("    double block %d: %.2f GB allocated", i,
                            ms$allocated_bytes$all$current / 1e9))
        }
        if (block_gc) {
            gc(verbose = FALSE)
        }
    }

    # The reference concatenates [text; image] inside every single block
    # and splits after; concatenating once here is numerically identical
    txt_len <- encoder_hidden_states$shape[2]
    hidden_states <- torch::torch_cat(
                                      list(encoder_hidden_states, hidden_states),
                                      dim = 2L
    )
    for (i in seq_along(self$single_transformer_blocks)) {
        hidden_states <- self$single_transformer_blocks[[i]](
                                                             hidden_states = hidden_states,
                                                             temb = temb,
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
