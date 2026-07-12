% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_ltx_audio_vae_decode}
\alias{yq_ltx_audio_vae_decode}
\title{LTX-2.3 audio VAE decoder forward (anvl)}
\usage{
yq_ltx_audio_vae_decode(z, w)
}
\arguments{
\item{z}{AnvlArray \code{[B, C_lat, T, M]} unpacked audio latents.}

\item{w}{Weights pytree from \code{\link{yq_ltx_audio_vae_load_weights}}.}
}
\value{
AnvlArray \code{[B, out_ch, 4T-3, mel_bins]} mel spectrogram
  (\code{4T-3} frames in causal mode).
}
\description{
\code{anvl::jit()} the wrapper \code{function(z) yq_ltx_audio_vae_decode(z, w)}.
\code{z} must be the unpacked decoder input \code{[B, C_lat, T, M]}
(see \code{\link{yq_ltx_audio_vae_prepare}}). All crop/pad geometry is
static (derived from \code{z}'s trace-time shape).
}
