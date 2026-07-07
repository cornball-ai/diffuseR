% tinyrox says don't edit this manually, but it can't stop you!
\name{flux2_unpack_latents_with_ids}
\alias{flux2_unpack_latents_with_ids}
\title{Unpack FLUX.2 tokens back to a latent grid via position ids}
\usage{
flux2_unpack_latents_with_ids(x, ids, height, width)
}
\arguments{
\item{x}{Tensor [B, S, C] of tokens.}

\item{ids}{Tensor [S, 4] (or [B, S, 4]) of position ids.}

\item{height,width}{Integers. Packed grid dimensions.}
}
\value{
Tensor [B, C, height, width].
}
\description{
Scatters tokens to (H, W) positions taken from the id columns
(H = column 2, W = column 3, 0-based values). Reference:
Flux2KleinPipeline._unpack_latents_with_ids.
}
