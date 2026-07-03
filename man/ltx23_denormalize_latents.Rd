% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_denormalize_latents}
\alias{ltx23_denormalize_latents}
\title{Denormalize latents with the VAE's per-channel statistics}
\usage{
ltx23_denormalize_latents(latents, latents_mean, latents_std)
}
\arguments{
\item{latents}{Tensor [B, C, F, H, W].}

\item{latents_mean,latents_std}{Tensors [C].}
}
\value{
Denormalized latents ready for the decoder.
}
\description{
Denormalize latents with the VAE's per-channel statistics
}
