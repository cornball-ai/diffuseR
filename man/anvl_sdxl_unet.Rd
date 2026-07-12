% tinyrox says don't edit this manually, but it can't stop you!
\name{anvl_sdxl_unet}
\alias{anvl_sdxl_unet}
\title{Stable Diffusion XL UNet (anvl port of unet_sdxl_native)}
\description{
anvl re-implementation of \code{diffuseR::unet_sdxl_native}: the SDXL
UNet2DConditionModel. \code{conv_in}; three down blocks (two ResNet
blocks each; the first is plain ResNet + stride-2 downsample, the
second and third add a SpatialTransformer of depth 2 and 10 after
every ResNet, and the third has no downsample); the mid block (ResNet,
depth-10 SpatialTransformer, ResNet); three up blocks (three ResNet
blocks each, concatenating the LIFO down-path skips before every
ResNet, cross-attention on the first two blocks with depth 10 and 2,
and a nearest-2x upsample conv on all but the last); the sinusoidal
timestep embedding + MLP; the SDXL added-conditioning path (pooled
text embeds concatenated with the sinusoidal time-ids, projected by a
two-layer MLP and added to the timestep embedding); and the
GroupNorm-32 -> SiLU -> conv output head. The SpatialTransformer is
GroupNorm -> linear proj_in -> N BasicTransformerBlocks
(LayerNorm->self-attn, LayerNorm->cross-attn to the text embeddings,
LayerNorm->GEGLU feed-forward, each residual) -> linear proj_out +
residual.
}
\details{
\code{cross_attention_dim} is the concatenated CLIP-L + OpenCLIP
bigG text width (2048); \code{attention_head_dim} is 64; the
transformer depth is 0/2/10 down, 10 mid, 10/2/0 up. SDXL uses
linear (not 1x1-conv) proj_in / proj_out.

The conv helpers (\code{yq_conv3x3}, \code{yq_conv1x1},
\code{yq_add_conv_bias}) are shared with the VAE port
(\code{R/anvl_vae.R}).

}
