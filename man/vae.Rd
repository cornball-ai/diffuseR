% tinyrox says don't edit this manually, but it can't stop you!
\name{vae}
\alias{vae}
\title{FLUX.2 VAE decoder (anvl port of AutoencoderKLFlux2)}
\description{
anvl re-implementation of \code{diffuseR::flux2_vae_decoder}: the
standard AutoencoderKL decoder (conv_in, mid block with self
attention, four up blocks of three ResNet blocks with nearest-2x
upsampling on the first three, GroupNorm output head) plus the FLUX.2
\code{post_quant_conv} and BatchNorm latent de-normalization. Decodes
\code{[B, 32, H, W]} latents to \code{[B, 3, 8H, 8W]} pixels in
[-1, 1].
}
