% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_sdxl_vae_prepare}
\alias{yq_sdxl_vae_prepare}
\title{Prepare SDXL latents for the VAE decoder (scaling factor)}
\usage{
yq_sdxl_vae_prepare(z, scaling_factor = .YQ_SDXL_VAE_SCALING)
}
\arguments{
\item{z}{AnvlArray \code{[B, 4, H, W]} latents from the sampling loop.}

\item{scaling_factor}{Numeric. VAE scaling factor (SDXL: 0.13025).}
}
\value{
AnvlArray \code{[B, 4, H, W]}, rescaled.
}
\description{
The diffusers SDXL pipeline divides latents by the VAE
\code{scaling_factor} (0.13025) before decoding
(\code{z = z / scaling_factor}); this applies that rescale, returning
the input to \code{\link{yq_sdxl_vae_decode}}. A scalar-broadcast
divide, so latents stay on device (no host round-trip).
}
