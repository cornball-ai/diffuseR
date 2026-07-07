% tinyrox says don't edit this manually, but it can't stop you!
\name{flux2_pack_latents}
\alias{flux2_pack_latents}
\title{Pack patchified FLUX.2 latents into tokens}
\usage{
flux2_pack_latents(latents)
}
\arguments{
\item{latents}{Tensor [B, C, H, W].}
}
\value{
Tensor [B, H * W, C].
}
\description{
[B, C, H, W] -> [B, H * W, C] (row-major spatial flatten,
channels-last). Reference: Flux2KleinPipeline._pack_latents.
}
