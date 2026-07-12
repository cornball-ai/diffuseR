% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_ltx_vae_prepare}
\alias{yq_ltx_vae_prepare}
\title{Prepare LTX latents for the VAE decoder (per-channel de-normalization)}
\usage{
yq_ltx_vae_prepare(z, latents_mean, latents_std)
}
\arguments{
\item{z}{AnvlArray \code{[B, C, F, H, W]} normalized latents.}

\item{latents_mean,latents_std}{Numeric vectors \code{[C]}.}
}
\value{
AnvlArray \code{[B, C, F, H, W]}, de-normalized.
}
\description{
Applies the inverse of the VAE's per-channel latent normalization,
\code{z * std + mean} broadcast over \code{[B, C, F, H, W]} (mirrors
\code{diffuseR::ltx23_denormalize_latents}). The stats are supplied as
host-side R vectors; the affine runs device-side so \code{z} stays on
the device. The checkpoint's \code{scaling_factor} is 1.0, so this
per-channel affine is the whole de-normalization.
}
