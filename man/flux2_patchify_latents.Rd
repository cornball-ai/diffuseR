% tinyrox says don't edit this manually, but it can't stop you!
\name{flux2_patchify_latents}
\alias{flux2_patchify_latents}
\title{Patchify FLUX.2 latents (2x2 -> channels)}
\usage{
flux2_patchify_latents(latents)
}
\arguments{
\item{latents}{Tensor [B, C, H, W]; H and W must be even.}
}
\value{
Tensor [B, C * 4, H / 2, W / 2].
}
\description{
[B, C, H, W] -> [B, 4C, H/2, W/2], channel order (C, ph, pw).
Reference: Flux2KleinPipeline._patchify_latents.
}
