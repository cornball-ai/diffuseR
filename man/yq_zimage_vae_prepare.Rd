% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_zimage_vae_prepare}
\alias{yq_zimage_vae_prepare}
\title{Prepare Z-Image / FLUX.1 latents for the VAE decoder (shift + scale)}
\usage{
yq_zimage_vae_prepare(z, scaling_factor = .YQ_ZIMAGE_VAE_SCALING,
                      shift_factor = .YQ_ZIMAGE_VAE_SHIFT)
}
\arguments{
\item{z}{AnvlArray \code{[B, 16, H, W]} latents from the sampling loop.}

\item{scaling_factor}{Numeric. VAE scaling factor (FLUX.1: 0.3611).}

\item{shift_factor}{Numeric. VAE shift factor (FLUX.1: 0.1159).}
}
\value{
AnvlArray \code{[B, 16, H, W]}, de-normalized.
}
\description{
The diffusers FLUX pipeline de-normalizes latents before decoding with
a scalar affine (\code{z = z / scaling_factor + shift_factor}); this
applies that, returning the input to \code{\link{yq_zimage_vae_decode}}.
Both are 0-d scalar broadcasts, so latents stay on device (no host
round-trip). This is the FLUX.1 16-channel VAE convention, shared
verbatim by Z-Image-Turbo (\code{scaling_factor = 0.3611},
\code{shift_factor = 0.1159}) — distinct from SD 2.1's single 0.18215
divide and from FLUX.2's per-channel BatchNorm de-normalization.
}
