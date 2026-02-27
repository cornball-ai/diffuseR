% tinyrox says don't edit this manually, but it can't stop you!
\name{upsample_video_latents}
\alias{upsample_video_latents}
\title{Upsample Video Latents}
\usage{
upsample_video_latents(
  latents,
  upsampler,
  latents_mean,
  latents_std,
  device = NULL,
  dtype = NULL
)
}
\arguments{
\item{latents}{Tensor. Latent tensor \code{[B, C, T, H, W]}.}

\item{upsampler}{A \code{latent_upsampler} module.}

\item{latents_mean}{Tensor. Per-channel mean (from VAE).}

\item{latents_std}{Tensor. Per-channel std (from VAE).}

\item{device}{Character. Device for computation.}

\item{dtype}{Torch dtype for computation.}
}
\value{
Upsampled latent tensor \code{[B, C, T, 2H, 2W]}.
}
\description{
Un-normalizes latents using VAE per-channel statistics, runs through
the upsampler, then re-normalizes.
}
\keyword{internal}
