% tinyrox says don't edit this manually, but it can't stop you!
\name{load_flux2_vae_encoder}
\alias{load_flux2_vae_encoder}
\title{Load the FLUX.2 VAE encoder from safetensors}
\usage{
load_flux2_vae_encoder(
  path,
  latent_channels = 32L,
  block_channels = c(128L, 256L, 512L, 512L),
  layers_per_block = 2L,
  norm_groups = 32L,
  verbose = TRUE
)
}
\arguments{
\item{path}{Path to the VAE .safetensors file (or a directory
containing diffusion_pytorch_model.safetensors).}

\item{verbose}{Logical.}

\item{latent_channels,block_channels,layers_per_block,norm_groups}{Constructor arguments for \code{\link{flux2_vae_encoder}}.}
}
\value{
The loaded \code{flux2_vae_encoder} in eval mode.
}
\description{
Loads the encoder half plus quant_conv and the BatchNorm running
statistics; decoder and post_quant_conv keys are skipped.
}
