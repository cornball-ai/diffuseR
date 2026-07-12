% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_sd_unet}
\alias{yq_sd_unet}
\title{Stable Diffusion 2.1 UNet forward (anvl)}
\usage{
yq_sd_unet(block_out_channels = c(320L, 640L, 1280L, 1280L),
           layers_per_block = 2L, cross_attention_dim = 1024L,
           attention_head_dim = 64L)
}
\arguments{
\item{block_out_channels}{Integer vector. Channels per block.}

\item{layers_per_block}{Integer. ResNet blocks per down/up block.}

\item{cross_attention_dim}{Integer. Text-encoder context width.}

\item{attention_head_dim}{Integer. Per-head dimension.}
}
\value{
Function of \code{(sample, t_sin, context, w)}:
  \itemize{
    \item sample \code{[B, 4, H, W]} noisy latents
    \item t_sin \code{[B, block_out_channels[1]]} sinusoidal timestep
      embedding (see \code{\link{yq_sd_time_embed}})
    \item context \code{[B, seq, cross_attention_dim]} text embeddings
    \item w weights pytree (\code{\link{yq_sd_unet_load_weights}})
  }
  returning the predicted noise \code{[B, 4, H, W]}.
}
\description{
Returns a closure over the static SD 2.1 configuration;
\code{anvl::jit()} the closure. All array arguments are dynamic inputs;
weights travel as the pytree from \code{\link{yq_sd_unet_load_weights}}.
}
