#' FLUX 3 Transformer
#'
#' Fresh R port of Flux3Transformer2DModel from the diffusers reference
#' implementation (Apache-2.0,
#' src/diffusers/models/transformers/transformer_flux3.py). Defaults are
#' the FLUX 3 Image Klein configuration: 5 per-mode blocks for each of
#' the image, reference-image and text modes, then 28 joint blocks.
#' Conditioning is a per-token timestep embedding (the image tokens at
#' the flow timestep, text and reference tokens at 0); the step-distilled
#' (flow-map) checkpoint adds the embedding of the timestep each step
#' jumps to.
#'
#' @name dit_flux3
NULL

# Linear(dim, 3 dim) on an already SiLU-activated embedding. Reference:
# Flux3Modulation.
flux3_modulation <- torch::nn_module(
                                     "flux3_modulation",
                                     initialize = function(dim) {
    self$linear <- linear_noinit(dim, dim * 3L, bias = FALSE)
},
                                     forward = function(temb_act) {
    self$linear(temb_act)
}
)

#' FLUX 3 transformer model
#'
#' The forward mirrors the reference: per-mode blocks for the text,
#' image and (optional) reference tokens, each with its own rotary
#' embedding, then joint blocks over [text; image; reference] and an
#' output projection of the image tokens only. Modulation is computed
#' once per mode on the [B, 1, dim] embedding and broadcast over the
#' mode's tokens (the reference expands to [B, S, dim] first; every row
#' is identical).
#'
#' @param in_channels Integer. Packed latent channels (128).
#' @param cond_in_channels Integer. Reference-image token channels (128).
#' @param num_layers Integer. Per-mode blocks per mode (5).
#' @param num_single_layers Integer. Joint blocks (28).
#' @param attention_head_dim Integer. Per-head dimension (128).
#' @param num_attention_heads Integer. Attention heads (24).
#' @param joint_attention_dim Integer. Text embedding dim (20480).
#' @param timestep_channels Integer. Sinusoid width (256).
#' @param mlp_ratio Numeric. Feed-forward multiplier (3.0).
#' @param axes_dims_rope Integer vector. Per-axis rotary dims.
#' @param rope_theta Numeric. Rotary base frequency (10000).
#' @param eps Numeric. Norm epsilon.
#' @param flow_map Logical. Step-distilled checkpoint: adds
#'   \code{target_timestep_embedder} and requires \code{target_timestep}.
#'
#' @return Module whose forward(hidden_states, encoder_hidden_states,
#'   timestep, img_ids, txt_ids, image_cond_hidden_states, image_cond_ids,
#'   target_timestep, chunk_size) returns the predicted velocity of the
#'   image tokens [B, S_img, in_channels]. Timesteps are flow times in
#'   [0, 1]; they are scaled by 1000 internally.
#'
#' @export
flux3_transformer <- torch::nn_module(
                                      "flux3_transformer",
                                      initialize = function(in_channels = 128L,
        cond_in_channels = 128L,
        num_layers = 5L,
        num_single_layers = 28L,
        attention_head_dim = 128L,
        num_attention_heads = 24L,
        joint_attention_dim = 20480L,
        timestep_channels = 256L,
        mlp_ratio = 3.0,
        axes_dims_rope = c(32L, 32L, 32L, 32L),
        rope_theta = 10000,
        eps = 1e-6,
        flow_map = FALSE) {
    inner_dim <- num_attention_heads * attention_head_dim
    self$inner_dim <- inner_dim
    self$timestep_channels <- as.integer(timestep_channels)
    self$axes_dims_rope <- as.integer(axes_dims_rope)
    self$rope_theta <- rope_theta
    self$flow_map <- isTRUE(flow_map)

    self$timestep_embedder <- ltx23_timestep_embedding(timestep_channels,
        inner_dim, bias = FALSE)
    if (self$flow_map) {
        self$target_timestep_embedder <- ltx23_timestep_embedding(
            timestep_channels, inner_dim, bias = FALSE)
    }

    modes <- c("image", "image_cond", "txt")
    mods <- function() {
        torch::nn_module_dict(stats::setNames(
                lapply(modes, function(m) flux3_modulation(inner_dim)), modes))
    }
    self$early_stream_modulations <- mods()
    self$single_stream_modulations <- mods()

    self$x_embedders <- torch::nn_module_dict(list(
            image = linear_noinit(in_channels, inner_dim, bias = FALSE),
            image_cond = linear_noinit(cond_in_channels, inner_dim,
                                       bias = FALSE)))
    self$context_embedder <- linear_noinit(joint_attention_dim,
        inner_dim, bias = FALSE)

    blocks <- function(n, ctor) {
        torch::nn_module_list(lapply(seq_len(n), function(i) {
            ctor(inner_dim, num_attention_heads, attention_head_dim,
                 mlp_ratio = mlp_ratio, eps = eps)
        }))
    }
    self$mode_blocks <- torch::nn_module_dict(list(
            image = blocks(num_layers, flux3_mode_block),
            image_cond = blocks(num_layers, flux3_mode_block)))
    self$txt_mode_blocks <- blocks(num_layers, flux3_mode_block)
    self$single_transformer_blocks <- blocks(num_single_layers,
        flux3_single_block)

    self$norm_out <- flux_ada_layer_norm_continuous(inner_dim, inner_dim,
        bias = FALSE)
    self$proj_out <- linear_noinit(inner_dim, in_channels, bias = FALSE)
},
                                      embed_timestep = function(embedder, t, dtype) {
    proj <- ltx23_get_timestep_embedding(t$to(dtype = dtype)$mul(1000),
        self$timestep_channels, flip_sin_to_cos = TRUE,
        downscale_freq_shift = 0)
    embedder(proj$to(dtype = dtype))
},
                                      forward = function(hidden_states, encoder_hidden_states, timestep,
        img_ids, txt_ids, image_cond_hidden_states = NULL,
        image_cond_ids = NULL, target_timestep = NULL,
        chunk_size = NULL) {
    if (is.null(image_cond_hidden_states) != is.null(image_cond_ids)) {
        stop("`image_cond_hidden_states` and `image_cond_ids` go together.")
    }
    if (is.null(target_timestep) == self$flow_map) {
        stop("`target_timestep` is required by, and only by, a flow-map ",
             "transformer (flow_map = TRUE).")
    }
    dtype <- hidden_states$dtype
    batch_size <- hidden_states$shape[1]
    image_seq_len <- hidden_states$shape[2]
    has_cond <- !is.null(image_cond_hidden_states)

    # 1. Timestep embeddings [B, dim]: image tokens at `timestep`, text
    #    and reference tokens at 0; the flow-map target is added to both
    image_temb <- self$embed_timestep(self$timestep_embedder, timestep, dtype)
    zero_temb <- self$embed_timestep(self$timestep_embedder,
                                     torch::torch_zeros(batch_size, device = hidden_states$device), dtype)
    if (self$flow_map) {
        target_temb <- self$embed_timestep(self$target_timestep_embedder,
            target_timestep, dtype)
        image_temb <- image_temb + target_temb
        zero_temb <- zero_temb + target_temb
    }
    zero_act <- torch::nnf_silu(zero_temb)$unsqueeze(2L) # [B, 1, dim]
    image_act <- torch::nnf_silu(image_temb)$unsqueeze(2L)

    # 2. Text context
    txt_early <- self$early_stream_modulations$txt(zero_act)
    txt_single <- self$single_stream_modulations$txt(zero_act)
    encoder_hidden_states <- self$context_embedder(encoder_hidden_states)
    txt_rope <- flux_pos_embed(txt_ids, self$axes_dims_rope, self$rope_theta)

    # 3. Image tokens
    image_early <- self$early_stream_modulations$image(image_act)
    image_single <- self$single_stream_modulations$image(image_act)
    hidden_states <- self$x_embedders$image(hidden_states)
    image_rope <- flux_pos_embed(img_ids, self$axes_dims_rope, self$rope_theta)

    # 4. Reference tokens (optional)
    if (has_cond) {
        cond_early <- self$early_stream_modulations$image_cond(zero_act)
        cond_single <- self$single_stream_modulations$image_cond(zero_act)
        image_cond_hidden_states <- self$x_embedders$image_cond(
            image_cond_hidden_states)
        cond_rope <- flux_pos_embed(image_cond_ids, self$axes_dims_rope,
                                    self$rope_theta)
    }

    # 5. Per-mode blocks (each mode attends within itself only)
    block_gc <- isTRUE(getOption("diffuseR.block_gc"))
    for (i in seq_along(self$txt_mode_blocks)) {
        encoder_hidden_states <- self$txt_mode_blocks[[i]](
                                      encoder_hidden_states, txt_early, txt_rope, chunk_size)
    }
    for (i in seq_along(self$mode_blocks$image)) {
        hidden_states <- self$mode_blocks$image[[i]](
                                      hidden_states, image_early, image_rope, chunk_size)
        if (block_gc) {
            gc(verbose = FALSE, full = FALSE)
        }
    }
    if (has_cond) {
        for (i in seq_along(self$mode_blocks$image_cond)) {
            image_cond_hidden_states <- self$mode_blocks$image_cond[[i]](
                                      image_cond_hidden_states, cond_early, cond_rope, chunk_size)
            if (block_gc) {
                gc(verbose = FALSE, full = FALSE)
            }
        }
    }

    # 6. Joint blocks over [txt, image, image_cond]
    if (has_cond) {
        cond_len <- image_cond_hidden_states$shape[2]
        content <- torch::torch_cat(list(hidden_states,
                image_cond_hidden_states), dim = 2L)
        content_mod <- torch::torch_cat(list(
                image_single$expand(c(-1L, image_seq_len, -1L)),
                cond_single$expand(c(-1L, cond_len, -1L))), dim = 2L)
        ropes <- list(txt_rope, image_rope, cond_rope)
    } else {
        content <- hidden_states
        content_mod <- image_single
        ropes <- list(txt_rope, image_rope)
    }
    joint_rope <- list(
                       torch::torch_cat(lapply(ropes, `[[`, 1L), dim = 1L),
                       torch::torch_cat(lapply(ropes, `[[`, 2L), dim = 1L))
    rm(hidden_states, image_cond_hidden_states)

    for (i in seq_along(self$single_transformer_blocks)) {
        res <- self$single_transformer_blocks[[i]](
                                      hidden_states = content,
                                      encoder_hidden_states = encoder_hidden_states,
                                      temb_mod = content_mod,
                                      temb_mod_context = txt_single,
                                      image_rotary_emb = joint_rope,
                                      chunk_size = chunk_size)
        encoder_hidden_states <- res[[1]]
        content <- res[[2]]
        if (block_gc) {
            gc(verbose = FALSE, full = FALSE)
        }
    }

    # 7. Output: the image tokens sit first in the content sequence
    out <- content$narrow(2L, 1L, image_seq_len)
    self$proj_out(self$norm_out(out, image_temb))
}
)
