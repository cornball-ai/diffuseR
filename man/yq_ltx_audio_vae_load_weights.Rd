% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_ltx_audio_vae_load_weights}
\alias{yq_ltx_audio_vae_load_weights}
\title{Load LTX-2.3 audio VAE decoder weights into an anvl pytree}
\usage{
yq_ltx_audio_vae_load_weights(path, device = "cpu", strict = TRUE,
                              mel_bins = 64L, causality_axis = "height")
}
\arguments{
\item{path}{Path to the decoder state_dict \code{.safetensors} (native
\code{ltx23_audio_decoder} key names, no \code{decoder.} prefix).}

\item{device}{Character. Target device.}

\item{strict}{Logical. Enforce the exact key census.}

\item{mel_bins}{Integer. Output mel bins (crop/pad target).}

\item{causality_axis}{Character. Padding convention ("height" default).}
}
\value{
Weights pytree for \code{\link{yq_ltx_audio_vae_decode}}.
}
\description{
Reads a \code{diffuseR::ltx23_audio_decoder} state_dict (F16/F32 upcast
to f32): \code{conv_in}, the two mid ResNet blocks, the up stages (each
a ResNet stack whose first block carries a \code{nin_shortcut}, plus an
optional \code{upsample} conv), and \code{conv_out}. The up stages are
returned in forward-processing order (top level first, i.e.
\code{up.<n-1>} down to \code{up.0}). The per-channel RMS norms carry no
weights, so the whole census is Conv2d \code{weight}/\code{bias} pairs.
Conv weights load raw \code{[out, in, k_h, k_w]}. Output channels are
read from \code{conv_out}. With \code{strict = TRUE} every file key must
be consumed exactly once.
}
