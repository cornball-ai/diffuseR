% tinyrox says don't edit this manually, but it can't stop you!
\name{load_flux2_vae_decoder}
\alias{load_flux2_vae_decoder}
\title{Load the FLUX.2 VAE decoder from safetensors}
\usage{
load_flux2_vae_decoder(path, latent_channels = 32L,
                       block_channels = c(512L, 512L, 256L, 128L),
                       norm_groups = 32L, verbose = TRUE)
}
\arguments{
\item{path}{Path to the VAE .safetensors file (or a directory
containing diffusion_pytorch_model.safetensors).}

\item{verbose}{Logical.}

\item{latent_channels,block_channels,norm_groups}{Constructor
arguments for \code{\link{flux2_vae_decoder}}.}
}
\value{
The loaded \code{flux2_vae_decoder} in eval mode.
}
\description{
Loads the decoder half plus post_quant_conv and the BatchNorm running
statistics; encoder and quant_conv keys are skipped (txt2img needs no
encoder).
}
