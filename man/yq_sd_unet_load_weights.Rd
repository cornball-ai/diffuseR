% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_sd_unet_load_weights}
\alias{yq_sd_unet_load_weights}
\title{Load SD 2.1 UNet weights into an anvl pytree}
\usage{
yq_sd_unet_load_weights(path, block_out_channels = c(320L, 640L, 1280L, 1280L),
                        layers_per_block = 2L, cross_attention_dim = 1024L,
                        attention_head_dim = 64L, device = "cpu", strict = TRUE)
}
\arguments{
\item{path}{Path to \code{unet/diffusion_pytorch_model.safetensors}.}

\item{device}{Character. Target device.}

\item{strict}{Logical. Enforce the exact key census.}

\item{block_out_channels,layers_per_block,cross_attention_dim,attention_head_dim}{Architecture (SD 2.1 defaults).}
}
\value{
Weights pytree for \code{\link{yq_sd_unet}}.
}
\description{
Reads the diffusers \code{UNet2DConditionModel} checkpoint (F16 upcast
to f32) and mirrors its key tree as a nested named list for
\code{\link{yq_sd_unet}}. Conv weights load raw \code{[out, in, kH, kW]}
(what \code{nv_conv2d} wants); linear / attention weights transpose to
\code{[in, out]} for \code{\link[yunque]{yq_linear}}. With
\code{strict = TRUE} every checkpoint key must be consumed exactly once
(and no key requested that the file lacks), so a wrong architecture or
a missed weight fails loudly.
}
