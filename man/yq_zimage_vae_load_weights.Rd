% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_zimage_vae_load_weights}
\alias{yq_zimage_vae_load_weights}
\title{Load Z-Image / FLUX.1 VAE decoder weights into an anvl pytree}
\usage{
yq_zimage_vae_load_weights(path, device = "cpu", strict = TRUE)
}
\arguments{
\item{path}{Path to the decoder state_dict \code{.safetensors} (native
\code{vae_decoder_native} key names, no \code{decoder.} prefix).}

\item{device}{Character. Target device.}

\item{strict}{Logical. Enforce the exact key census.}
}
\value{
Weights pytree for \code{\link{yq_zimage_vae_decode}}.
}
\description{
Reads the decode half of a \code{diffuseR::vae_decoder_native}
(\code{latent_channels = 16}) state_dict (F16/F32 upcast to f32):
\code{conv_in}, the mid block, the four up blocks, and the output head.
There is no \code{post_quant_conv} and no encoder in this checkpoint
(the FLUX VAE omits both). Conv weights load raw \code{[out, in, kH, kW]}
(what \code{\link[anvl]{nv_conv2d}} expects); the attention linears
transpose to \code{[in, out]} for \code{\link[yunque]{yq_linear}}.
Optional ResNet \code{conv_shortcut} and up-block \code{upsamplers} are
detected by key existence. With \code{strict = TRUE} every key in the
file must be consumed exactly once (and no requested key may be
absent), so a wrong architecture or a missed weight fails loudly.
}
