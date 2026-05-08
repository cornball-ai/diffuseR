% tinyrox says don't edit this manually, but it can't stop you!
\name{.get_vae_stats}
\alias{.get_vae_stats}
\title{Get VAE per-channel statistics}
\usage{
.get_vae_stats(model_dir = NULL, verbose = TRUE)
}
\description{
Loads latents_mean and latents_std from VAE config/weights for
normalize/denormalize operations in the upsampler.
}
\keyword{internal}
