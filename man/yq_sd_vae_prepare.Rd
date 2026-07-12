% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_sd_vae_prepare}
\alias{yq_sd_vae_prepare}
\title{Prepare SD 2.1 latents for the VAE decoder (scaling factor)}
\usage{
yq_sd_vae_prepare(z, scaling_factor = .YQ_SD_VAE_SCALING)
}
\arguments{
\item{z}{AnvlArray \code{[B, 4, H, W]} latents from the sampling loop.}

\item{scaling_factor}{Numeric. VAE scaling factor (SD 2.1: 0.18215).}
}
\value{
AnvlArray \code{[B, 4, H, W]}, rescaled.
}
\description{
The diffusers SD pipeline divides latents by the VAE
\code{scaling_factor} (0.18215) before decoding
(\code{z = z / scaling_factor}); this applies that rescale, returning
the input to \code{\link{yq_sd_vae_decode}}. A scalar-broadcast divide,
so latents stay on device (no host round-trip).
}
