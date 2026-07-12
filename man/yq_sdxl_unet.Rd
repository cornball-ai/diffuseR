% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_sdxl_unet}
\alias{yq_sdxl_unet}
\title{Stable Diffusion XL UNet forward (anvl)}
\usage{
yq_sdxl_unet(block_out_channels = c(320L, 640L, 1280L), layers_per_block = 2L,
             transformer_layers_per_block = c(0L, 2L, 10L),
             cross_attention_dim = 2048L, attention_head_dim = 64L)
}
\arguments{
\item{block_out_channels}{Integer vector. Channels per block.}

\item{layers_per_block}{Integer. ResNet blocks per down/up block.}

\item{transformer_layers_per_block}{Integer vector. Transformer depth
per block.}

\item{cross_attention_dim}{Integer. Text-encoder context width.}

\item{attention_head_dim}{Integer. Per-head dimension.}
}
\value{
Function of \code{(sample, t_sin, time_ids_sin, text_embeds,
  context, w)}:
  \itemize{
    \item sample \code{[B, 4, H, W]} noisy latents
    \item t_sin \code{[B, block_out_channels[1]]} sinusoidal timestep
      embedding (see \code{\link{yq_sdxl_time_embed}})
    \item time_ids_sin \code{[B, 6 * addition_time_embed_dim]}
      sinusoidal time-ids (see \code{\link{yq_sdxl_time_ids_embed}})
    \item text_embeds \code{[B, 1280]} pooled text embeds
    \item context \code{[B, seq, cross_attention_dim]} text embeddings
    \item w weights pytree (\code{\link{yq_sdxl_unet_load_weights}})
  }
  returning the predicted noise \code{[B, 4, H, W]}.
}
\description{
Returns a closure over the static SDXL configuration;
\code{anvl::jit()} the closure. All array arguments are dynamic
inputs; weights travel as the pytree from
\code{\link{yq_sdxl_unet_load_weights}}.
}
