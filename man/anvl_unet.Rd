% tinyrox says don't edit this manually, but it can't stop you!
\name{anvl_unet}
\alias{anvl_unet}
\title{Stable Diffusion 2.1 UNet (anvl port of unet_native)}
\description{
anvl re-implementation of \code{diffuseR::unet_native}: the SD 2.1
UNet2DConditionModel. \code{conv_in}; four down blocks (two ResNet
blocks each, the first three also carrying a depth-1 SpatialTransformer
cross-attention after every ResNet plus a stride-2 downsample conv);
the mid block (ResNet, SpatialTransformer, ResNet); four up blocks
(three ResNet blocks each, concatenating the down-path skip connections
before every ResNet, cross-attention on all but the first block, and a
nearest-2x upsample conv on all but the last); the sinusoidal timestep
embedding + MLP; and the GroupNorm-32 -> SiLU -> conv output head. The
SpatialTransformer is GroupNorm -> linear proj_in -> N
BasicTransformerBlocks (LayerNorm->self-attn, LayerNorm->cross-attn to
the text embeddings, LayerNorm->GEGLU feed-forward, each residual) ->
linear proj_out + residual.
}
\details{
\code{cross_attention_dim} is the OpenCLIP ViT-H text width (1024);
\code{attention_head_dim} is 64; the transformer depth is 1 per block.
SD 2.1 uses linear (not 1x1-conv) proj_in / proj_out
(\code{use_linear_projection = TRUE}).

The conv helpers (\code{yq_conv3x3}, \code{yq_conv1x1},
\code{yq_add_conv_bias}) are shared with the VAE port
(\code{R/anvl_vae.R}).

}
