% tinyrox says don't edit this manually, but it can't stop you!
\name{flux_pack_latents}
\alias{flux_pack_latents}
\title{Pack FLUX latents into a patch sequence}
\usage{
flux_pack_latents(latents)
}
\arguments{
\item{latents}{Tensor of shape [B, C, H, W]; H and W must be even.}
}
\value{
Tensor of shape [B, (H/2) * (W/2), C * 4].
}
\description{
Packs a [B, C, H, W] latent into 2x2 patches, giving a sequence
[B, (H/2) * (W/2), C * 4]. Reference: FluxPipeline._pack_latents.
}
