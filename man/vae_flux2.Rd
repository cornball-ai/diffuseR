% tinyrox says don't edit this manually, but it can't stop you!
\name{vae_flux2}
\alias{vae_flux2}
\title{FLUX.2 Latent Layout and VAE Helpers}
\description{
Fresh R port of the FLUX.2 latent packing chain from the diffusers
reference (Apache-2.0, src/diffusers/pipelines/flux2/
pipeline_flux2_klein.py). The 32-channel VAE latent is patchified
2x2 into 128 channels, normalized with the VAE's BatchNorm running
statistics (there is no scalar scaling/shift factor in FLUX.2), and
flattened to channels-last tokens for the transformer.
}
