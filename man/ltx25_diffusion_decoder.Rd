% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx25_diffusion_decoder}
\alias{ltx25_diffusion_decoder}
\title{LTX-2.5 diffusion video decoder module}
\usage{
ltx25_diffusion_decoder(
  latent_channels = 128L,
  out_channels = 3L,
  patch_size = 4L,
  head_dim = 64L,
  stage_channels = c(2048L, 1024L, 512L, 512L, 256L),
  stage_depths = c(4L, 6L, 4L, 2L, 8L),
  stage_kernels = list(c(3L, 7L, 7L), c(3L, 7L, 7L), c(3L, 5L, 5L), c(3L, 5L, 5L)),
  upsample_strides = list(c(1L, 2L, 2L), c(2L, 1L, 1L), c(2L, 2L, 2L), c(2L, 2L, 2L)),
  upsample_reductions = c(2L, 2L, 1L, 2L),
  stage5_kernel = c(11L, 11L, 11L),
  t_emb_dim = 384L
)
}
\arguments{
\item{latent_channels,out_channels,patch_size,head_dim,stage_channels,stage_depths,stage_kernels,upsample_strides,upsample_reductions,stage5_kernel,t_emb_dim}{Architecture (diffusers \code{LTX2VideoDiffusionDecoder3d}).}
}
\value{
An \code{nn_module}; \code{$decode(z)} takes denormalized
  latents [B, 128, F, H/32, W/32] and returns pixels [B, 3, 8(F-1)+1,
  H, W] in about [-1, 1].
}
\description{
The decoder half of LTX-2.5's default video VAE: a small diffusion
model that renders pixels from the same latents the convolutional
decoder reads. Built by \code{\link{ltx25_load_diffusion_decoder}};
arguments default to the shipped configuration.
}
