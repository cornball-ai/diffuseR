% tinyrox says don't edit this manually, but it can't stop you!
\name{zimage_transformer}
\alias{zimage_transformer}
\title{Z-Image Transformer}
\usage{
zimage_transformer(in_channels = 16L, dim = 3840L, n_layers = 30L,
                   n_refiner_layers = 2L, n_heads = 30L, norm_eps = 1e-05,
                   cap_feat_dim = 2560L, rope_theta = 256, t_scale = 1000,
                   axes_dims = c(32L, 48L, 48L), patch_size = 2L,
                   f_patch_size = 1L)
}
\arguments{
\item{in_channels}{Integer. Latent channels. Default 16.}

\item{dim}{Integer. Model width. Default 3840.}

\item{n_layers}{Integer. Main trunk depth. Default 30.}

\item{n_refiner_layers}{Integer. Refiner depth. Default 2.}

\item{n_heads}{Integer. Attention heads. Default 30.}

\item{norm_eps}{Numeric. RMSNorm epsilon. Default 1e-5.}

\item{cap_feat_dim}{Integer. Caption embedding width. Default 2560.}

\item{rope_theta}{Numeric. RoPE base frequency. Default 256.}

\item{t_scale}{Numeric. Timestep scale. Default 1000.}

\item{axes_dims}{Integer vector. Per-axis rotary dims. Default
c(32, 48, 48).}

\item{patch_size}{Integer. Spatial patch size. Default 2.}

\item{f_patch_size}{Integer. Temporal patch size. Default 1.}
}
\description{
Fresh R port of ZImageTransformer2DModel from the diffusers reference
(Apache-2.0, src/diffusers/models/transformers/transformer_z_image.py).
Single-stream DiT: image tokens pass through a modulated noise
refiner, caption tokens through an unmodulated context refiner, then
both are concatenated (image first) and run through the main trunk.
The module tree mirrors the reference state-dict keys 1:1
(all_x_embedder.2-1, noise_refiner.N, context_refiner.N, layers.N,
all_final_layer.2-1, t_embedder, cap_embedder, x_pad_token,
cap_pad_token).
}
\details{
This port is batch-of-1: \code{x} is a single latent [C, F, H, W] and
\code{cap_feats} a single caption [L, cap_feat_dim], so sub-sequences
are uniform and no attention mask is needed. Padding to a multiple of
32 tokens uses the learned pad parameters, appended after embedding
(the reference pads raw features with repeats, embeds pointwise, then
overwrites the pad rows with the same learned tokens).

}
