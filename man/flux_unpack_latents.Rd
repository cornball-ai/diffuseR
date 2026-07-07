% tinyrox says don't edit this manually, but it can't stop you!
\name{flux_unpack_latents}
\alias{flux_unpack_latents}
\title{Unpack a FLUX patch sequence back into latents}
\usage{
flux_unpack_latents(latents, height, width, vae_scale_factor = 8L)
}
\arguments{
\item{latents}{Tensor of shape [B, S, C_packed].}

\item{vae_scale_factor}{Integer. Spatial downsampling of the VAE (8).}

\item{height,width}{Integers. Image height/width in pixels.}
}
\value{
Tensor of shape [B, C_packed / 4, height / 8, width / 8].
}
\description{
Inverse of \code{flux_pack_latents}. Height and width are the target
image dimensions in pixels; the latent grid is derived via the VAE
scale factor and the 2x2 patch size. Reference:
FluxPipeline._unpack_latents.
}
