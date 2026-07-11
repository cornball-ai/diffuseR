% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_flux2_vae_decode}
\alias{yq_flux2_vae_decode}
\title{FLUX.2 VAE decoder forward (anvl)}
\usage{
yq_flux2_vae_decode(z, w)
}
\arguments{
\item{z}{AnvlArray \code{[B, 32, H, W]} latents (already
BN-denormalized; see \code{\link{yq_flux2_vae_prepare}}).}

\item{w}{VAE weights pytree from \code{\link{yq_flux2_load_vae}}.}
}
\value{
AnvlArray \code{[B, 3, 8H, 8W]} pixels in [-1, 1].
}
\description{
FLUX.2 VAE decoder forward (anvl)
}
