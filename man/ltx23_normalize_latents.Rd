% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_normalize_latents}
\alias{ltx23_normalize_latents}
\title{Normalize latents with the VAE's per-channel statistics}
\usage{
ltx23_normalize_latents(latents, latents_mean, latents_std)
}
\arguments{
\item{latents}{Tensor [B, C, F, H, W].}

\item{latents_mean,latents_std}{Tensors [C].}
}
\value{
Normalized latents.
}
\description{
Normalize latents with the VAE's per-channel statistics
}
