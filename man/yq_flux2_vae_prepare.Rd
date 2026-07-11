% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_flux2_vae_prepare}
\alias{yq_flux2_vae_prepare}
\title{Prepare DiT latents for the FLUX.2 VAE decoder (host-side)}
\usage{
yq_flux2_vae_prepare(latents, h2, w2, bn_mean, bn_var, eps = 1e-04,
                     device = "cpu")
}
\arguments{
\item{latents}{R array \code{[B, S, 128]} (DiT output).}

\item{eps}{Numeric. BatchNorm epsilon (1e-4).}

\item{device}{Character.}

\item{h2,w2}{Integers. Packed grid dims (\code{S == h2 * w2}).}

\item{bn_mean,bn_var}{R vectors \code{[128]} of BatchNorm stats.}
}
\value{
AnvlArray \code{[B, 32, 2 h2, 2 w2]}, f32.
}
\description{
Turns the DiT's final packed latents into the decoder's input:
unpacks tokens to the grid (\code{[B, S, 128] -> [B, 128, h2, w2]}),
applies the inverse BatchNorm normalization on the 128 packed
channels (\code{z * sqrt(var + eps) + mean}), then un-patchifies the
2x2 blocks (\code{128 -> 32}, doubling each spatial dim). Parameter-
free reshaping and per-channel affine, computed in base R.
}
