% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_audio_decoder}
\alias{ltx23_audio_decoder}
\title{LTX-2.3 audio VAE decoder}
\usage{
ltx23_audio_decoder(base_channels = 128L, output_channels = 2L,
                    num_res_blocks = 2L, latent_channels = 8L,
                    ch_mult = c(1L, 2L, 4L), causality_axis = "height",
                    mel_bins = 64L)
}
\arguments{
\item{base_channels}{Integer.}

\item{output_channels}{Integer. Audio channels (2 = stereo).}

\item{num_res_blocks}{Integer. Per-level ResNet count (a stage runs
\code{num_res_blocks + 1} blocks).}

\item{latent_channels}{Integer.}

\item{ch_mult}{Integer vector. Channel multipliers per level.}

\item{causality_axis}{Character.}

\item{mel_bins}{Integer. Output mel bins (crop/pad target).}
}
\value{
Module whose forward(x) returns the decoded mel
  spectrogram reconstructed from an audio latent.
}
\description{
Latents [B, 8, L, 16] -> mel spectrogram [B, 2, 4L - 3, 64].
}
