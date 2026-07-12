% tinyrox says don't edit this manually, but it can't stop you!
\name{anvl_sd_vae}
\alias{anvl_sd_vae}
\title{Stable Diffusion 2.1 VAE decoder (anvl port of AutoencoderKL decode)}
\description{
anvl re-implementation of the SD 2.1 \code{AutoencoderKL} decode path:
the scalar \code{scaling_factor} latent rescale (host-side, see
\code{\link{yq_sd_vae_prepare}}), the 1x1 \code{post_quant_conv}, then
the standard AutoencoderKL decoder body (\code{conv_in}; a mid block of
ResNet, self attention, ResNet; four up blocks of three ResNet blocks
with nearest-2x upsampling on the first three; a GroupNorm-32 -> SiLU ->
\code{conv_out} output head). The architecture is identical to the
FLUX.2 VAE decoder apart from the 4-channel latent and the scalar
(rather than BatchNorm) latent normalization, so the block helpers
(\code{.yq_vae_resnet}, \code{.yq_vae_attention}, \code{.yq_vae_up_block})
and conv helpers are shared with \code{R/anvl_vae.R}. Decodes
\code{[B, 4, H, W]} latents to \code{[B, 3, 8H, 8W]} pixels in [-1, 1].
}
