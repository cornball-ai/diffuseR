% tinyrox says don't edit this manually, but it can't stop you!
\name{flux2_vae_decoder}
\alias{flux2_vae_decoder}
\title{FLUX.2 VAE decoder}
\usage{
flux2_vae_decoder(latent_channels = 32L,
                  block_channels = c(512L, 512L, 256L, 128L), norm_groups = 32L)
}
\arguments{
\item{latent_channels}{Integer (32 for FLUX.2).}

\item{block_channels}{Decoder block channels (reversed encoder
block_out_channels).}

\item{norm_groups}{Integer. Group norm groups.}
}
\value{
Module whose forward(z) decodes [B, 32, H, W] latents to
  [B, 3, 8H, 8W] images; \code{$bn$running_mean} /
  \code{$bn$running_var} carry the normalization statistics.
}
\description{
The AutoencoderKLFlux2 decode path: post_quant_conv (1x1, 32
channels) followed by the standard AutoencoderKL decoder body
(reused from \code{\link{vae_decoder_native}}), plus the BatchNorm
running statistics used for latent (de)normalization. Reference:
src/diffusers/models/autoencoders/autoencoder_kl_flux2.py.
}
