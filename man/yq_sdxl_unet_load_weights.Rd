% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_sdxl_unet_load_weights}
\alias{yq_sdxl_unet_load_weights}
\title{Load SDXL UNet weights into an anvl pytree}
\usage{
yq_sdxl_unet_load_weights(path, block_out_channels = c(320L, 640L, 1280L),
                          layers_per_block = 2L,
                          transformer_layers_per_block = c(0L, 2L, 10L),
                          cross_attention_dim = 2048L,
                          attention_head_dim = 64L, device = "cpu",
                          strict = TRUE)
}
\arguments{
\item{path}{Path to the SDXL UNet safetensors (or the random-init
state_dict fixture).}

\item{device}{Character. Target device.}

\item{strict}{Logical. Enforce the exact key census.}

\item{block_out_channels,layers_per_block,transformer_layers_per_block,cross_attention_dim,attention_head_dim}{Architecture (SDXL defaults).}
}
\value{
Weights pytree for \code{\link{yq_sdxl_unet}}.
}
\description{
Reads a diffusers \code{UNet2DConditionModel} SDXL state_dict (F16/F32
upcast to f32) and mirrors its key tree as a nested named list for
\code{\link{yq_sdxl_unet}}. Conv weights load raw \code{[out, in, kH,
kW]} (what \code{nv_conv2d} wants); linear / attention weights
transpose to \code{[in, out]} for \code{\link[yunque]{yq_linear}}.
With \code{strict = TRUE} every checkpoint key must be consumed
exactly once (and no key requested that the file lacks), so a wrong
architecture or a missed weight fails loudly.
}
