% tinyrox says don't edit this manually, but it can't stop you!
\name{flux2_unpatchify_latents}
\alias{flux2_unpatchify_latents}
\title{Unpatchify FLUX.2 latents (channels -> 2x2)}
\usage{
flux2_unpatchify_latents(latents)
}
\arguments{
\item{latents}{Tensor [B, 4C, H, W].}
}
\value{
Tensor [B, C, H * 2, W * 2].
}
\description{
Inverse of \code{\link{flux2_patchify_latents}}. Reference:
Flux2KleinPipeline._unpatchify_latents.
}
