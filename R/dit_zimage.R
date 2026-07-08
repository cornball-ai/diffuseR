#' Z-Image Transformer
#'
#' Fresh R port of ZImageTransformer2DModel from the diffusers reference
#' (Apache-2.0, src/diffusers/models/transformers/transformer_z_image.py).
#' Single-stream DiT: image tokens pass through a modulated noise
#' refiner, caption tokens through an unmodulated context refiner, then
#' both are concatenated (image first) and run through the main trunk.
#' The module tree mirrors the reference state-dict keys 1:1
#' (all_x_embedder.2-1, noise_refiner.N, context_refiner.N, layers.N,
#' all_final_layer.2-1, t_embedder, cap_embedder, x_pad_token,
#' cap_pad_token).
#'
#' This port is batch-of-1: \code{x} is a single latent [C, F, H, W] and
#' \code{cap_feats} a single caption [L, cap_feat_dim], so sub-sequences
#' are uniform and no attention mask is needed. Padding to a multiple of
#' 32 tokens uses the learned pad parameters, appended after embedding
#' (the reference pads raw features with repeats, embeds pointwise, then
#' overwrites the pad rows with the same learned tokens).
#'
#' @param in_channels Integer. Latent channels. Default 16.
#' @param dim Integer. Model width. Default 3840.
#' @param n_layers Integer. Main trunk depth. Default 30.
#' @param n_refiner_layers Integer. Refiner depth. Default 2.
#' @param n_heads Integer. Attention heads. Default 30.
#' @param norm_eps Numeric. RMSNorm epsilon. Default 1e-5.
#' @param cap_feat_dim Integer. Caption embedding width. Default 2560.
#' @param rope_theta Numeric. RoPE base frequency. Default 256.
#' @param t_scale Numeric. Timestep scale. Default 1000.
#' @param axes_dims Integer vector. Per-axis rotary dims. Default
#'   c(32, 48, 48).
#' @param patch_size Integer. Spatial patch size. Default 2.
#' @param f_patch_size Integer. Temporal patch size. Default 1.
#'
#' @export
zimage_transformer <- torch::nn_module(
                                       "zimage_transformer",
                                       initialize = function(in_channels = 16L, dim = 3840L, n_layers = 30L,
        n_refiner_layers = 2L, n_heads = 30L, norm_eps = 1e-5,
        cap_feat_dim = 2560L, rope_theta = 256, t_scale = 1000,
        axes_dims = c(32L, 48L, 48L), patch_size = 2L, f_patch_size = 1L) {
    stopifnot(dim %/% n_heads == sum(axes_dims))
    self$in_channels <- in_channels
    self$out_channels <- in_channels
    self$dim <- dim
    self$rope_theta <- rope_theta
    self$t_scale <- t_scale
    self$axes_dims <- axes_dims
    self$patch_size <- patch_size
    self$f_patch_size <- f_patch_size
    patch_key <- paste0(patch_size, "-", f_patch_size)
    patch_dim <- f_patch_size * patch_size * patch_size * in_channels

    embedders <- list(torch::nn_linear(patch_dim, dim, bias = TRUE))
    names(embedders) <- patch_key
    self$all_x_embedder <- torch::nn_module_dict(embedders)

    finals <- list(zimage_final_layer(dim, patch_dim))
    names(finals) <- patch_key
    self$all_final_layer <- torch::nn_module_dict(finals)

    self$noise_refiner <- torch::nn_module_list(lapply(
                                                       seq_len(n_refiner_layers),
                                                       function(i) zimage_block(dim, n_heads, norm_eps, modulation = TRUE)
    ))
    self$context_refiner <- torch::nn_module_list(lapply(
                                                         seq_len(n_refiner_layers),
                                                         function(i) zimage_block(dim, n_heads, norm_eps, modulation = FALSE)
    ))
    self$layers <- torch::nn_module_list(lapply(
                                                seq_len(n_layers),
                                                function(i) zimage_block(dim, n_heads, norm_eps, modulation = TRUE)
    ))

    self$t_embedder <- zimage_t_embedder(out_size = min(dim, 256L),
                                         mid_size = 1024L)
    self$cap_embedder <- torch::nn_sequential(
                                              ltx23_rms_norm(cap_feat_dim, eps = norm_eps),
                                              torch::nn_linear(cap_feat_dim, dim, bias = TRUE)
    )
    self$x_pad_token <- torch::nn_parameter(torch::torch_zeros(1L, dim))
    self$cap_pad_token <- torch::nn_parameter(torch::torch_zeros(1L, dim))
},
                                       forward = function(x, t, cap_feats, chunk_size = NULL) {
    # x: [C, F, H, W] latent; t: [1] in [0, 1]; cap_feats: [L, cap_feat_dim]
    device <- x$device
    p <- self$patch_size
    pf <- self$f_patch_size
    patch_key <- paste0(p, "-", pf)
    size <- x$shape[2:4]
    f_tokens <- size[1] %/% pf
    h_tokens <- size[2] %/% p
    w_tokens <- size[3] %/% p

    adaln <- self$t_embedder(t * self$t_scale)$to(dtype = x$dtype) # [1, 256]

    # Position ids and rotary frequencies
    cap_len <- cap_feats$shape[1]
    cap_padded <- cap_len + zimage_pad_len(cap_len)
    cap_freqs <- zimage_pos_embed(
                                  zimage_cap_pos_ids(cap_padded, device = device),
                                  axes_dim = self$axes_dims, theta = self$rope_theta
    )
    img_freqs <- zimage_pos_embed(
                                  zimage_img_pos_ids(h_tokens, w_tokens, start0 = cap_padded + 1L,
            f_tokens = f_tokens, device = device),
                                  axes_dim = self$axes_dims, theta = self$rope_theta
    )

    # Image tokens: patchify, embed, pad, refine
    tokens <- zimage_patchify(x, p, pf)
    img_len <- tokens$shape[1]
    x_emb <- self$all_x_embedder[[patch_key]](tokens)$unsqueeze(1L)
    img_pad <- zimage_pad_len(img_len)
    if (img_pad > 0L) {
        pad_rows <- self$x_pad_token$unsqueeze(1L)$expand(c(1L, img_pad, self$dim))
        x_emb <- torch::torch_cat(list(x_emb, pad_rows), dim = 2L)
    }
    for (i in seq_len(length(self$noise_refiner))) {
        x_emb <- self$noise_refiner[[i]](x_emb, img_freqs, adaln_input = adaln,
                                         chunk_size = chunk_size)
    }

    # Caption tokens: embed, pad, refine
    cap_emb <- self$cap_embedder(cap_feats)$unsqueeze(1L)
    if (cap_padded > cap_len) {
        pad_rows <- self$cap_pad_token$unsqueeze(1L)$expand(
                                                            c(1L, cap_padded - cap_len, self$dim)
        )
        cap_emb <- torch::torch_cat(list(cap_emb, pad_rows), dim = 2L)
    }
    for (i in seq_len(length(self$context_refiner))) {
        cap_emb <- self$context_refiner[[i]](cap_emb, cap_freqs,
                                             chunk_size = chunk_size)
    }

    # Unified sequence, image first
    unified <- torch::torch_cat(list(x_emb, cap_emb), dim = 2L)
    unified_freqs <- list(
                          torch::torch_cat(list(img_freqs[[1]], cap_freqs[[1]]), dim = 1L),
                          torch::torch_cat(list(img_freqs[[2]], cap_freqs[[2]]), dim = 1L)
    )
    for (i in seq_len(length(self$layers))) {
        unified <- self$layers[[i]](unified, unified_freqs,
                                    adaln_input = adaln,
                                    chunk_size = chunk_size)
    }

    out <- self$all_final_layer[[patch_key]](unified, adaln)
    zimage_unpatchify(out$squeeze(1L), size, p, pf, self$out_channels)
}
)
