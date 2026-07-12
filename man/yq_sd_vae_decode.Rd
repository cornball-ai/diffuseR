% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_sd_vae_decode}
\alias{yq_sd_vae_decode}
\title{Stable Diffusion 2.1 VAE decoder forward (anvl)}
\usage{
yq_sd_vae_decode(z, w)
}
\arguments{
\item{z}{AnvlArray \code{[B, 4, H, W]} latents, already scaling-factor
rescaled.}

\item{w}{VAE weights pytree from \code{\link{yq_sd_vae_load_weights}}.}
}
\value{
AnvlArray \code{[B, 3, 8H, 8W]} pixels in [-1, 1].
}
\description{
Reuses the shared AutoencoderKL block helpers from \code{R/anvl_vae.R};
\code{anvl::jit()} the wrapper \code{function(z) yq_sd_vae_decode(z, w)}.
\code{z} must already be rescaled by the VAE scaling factor (see
\code{\link{yq_sd_vae_prepare}}).
}
