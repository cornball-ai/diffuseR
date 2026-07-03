% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_adain_filter_latent}
\alias{ltx23_adain_filter_latent}
\title{Adaptive instance normalization between latent tensors}
\usage{
ltx23_adain_filter_latent(latents, reference_latents, factor = 1)
}
\arguments{
\item{latents}{Tensor [B, C, F, H, W].}

\item{reference_latents}{Tensor with the target statistics.}

\item{factor}{Numeric blend in [-10, 10]; 0 is identity.}
}
\value{
Filtered latents.
}
\description{
Matches each (batch, channel) slice's mean/std to the reference
latents, blended by \code{factor} (cf. diffusers
\code{LTX2LatentUpsamplePipeline.adain_filter_latent}).
}
