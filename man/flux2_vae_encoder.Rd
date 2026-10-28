% tinyrox says don't edit this manually, but it can't stop you!
\name{flux2_vae_encoder}
\alias{flux2_vae_encoder}
\title{FLUX.2 VAE encoder}
\usage{
flux2_vae_encoder(
  latent_channels = 32L,
  block_channels = c(128L, 256L, 512L, 512L),
  layers_per_block = 2L,
  norm_groups = 32L
)
}
\arguments{
\item{latent_channels}{Integer (32 for FLUX.2).}

\item{block_channels}{Encoder block_out_channels.}

\item{layers_per_block}{Integer. Resnets per down block.}

\item{norm_groups}{Integer. Group norm groups.}
}
\value{
Module whose forward(x) encodes [B, 3, H, W] images in
  [-1, 1] to [B, 32, H / 8, W / 8] latent means.
}
\description{
The AutoencoderKLFlux2 encode path: the standard diffusers AutoencoderKL
encoder body (conv_in, down blocks, attention mid block, GroupNorm +
SiLU, conv_out to 2 x latent channels) followed by quant_conv, plus the
BatchNorm running statistics used to normalize the patchified latent.
The forward returns the posterior mean (the reference pipeline encodes
with \code{sample_mode = "argmax"}). Reference:
src/diffusers/models/autoencoders/vae.py (Encoder) and
autoencoder_kl_flux2.py (Apache-2.0).
}
