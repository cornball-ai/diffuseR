% tinyrox says don't edit this manually, but it can't stop you!
\name{anvl_sdxl_vae}
\alias{anvl_sdxl_vae}
\title{Stable Diffusion XL VAE decoder (anvl port of AutoencoderKL decode)}
\description{
anvl re-implementation of the SDXL \code{AutoencoderKL} decode path.
SDXL's VAE is the same 4-channel \code{AutoencoderKL} as SD 2.1's
(only the \code{scaling_factor} differs: 0.13025 vs 0.18215), so the
decode body is identical: the scalar \code{scaling_factor} latent
rescale (host-side, see \code{\link{yq_sdxl_vae_prepare}}), the 1x1
\code{post_quant_conv}, then the standard AutoencoderKL decoder
(\code{conv_in}; a mid block of ResNet, self attention, ResNet; four up
blocks of three ResNet blocks with nearest-2x upsampling on the first
three; a GroupNorm-32 -> SiLU -> \code{conv_out} output head). The block
and conv helpers (\code{.yq_vae_resnet}, \code{.yq_vae_attention},
\code{.yq_vae_up_block}, \code{yq_conv3x3}, \code{yq_conv1x1}) are shared
with \code{R/anvl_vae.R}. Decodes \code{[B, 4, H, W]} latents to
\code{[B, 3, 8H, 8W]} pixels in [-1, 1].
}
