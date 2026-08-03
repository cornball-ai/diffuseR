% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_audio_encoder}
\alias{ltx23_audio_encoder}
\title{LTX-2.3 audio VAE encoder}
\usage{
ltx23_audio_encoder(base_channels = 128L, in_channels = 2L,
                    num_res_blocks = 2L, latent_channels = 8L,
                    ch_mult = c(1L, 2L, 4L), causality_axis = "height")
}
\arguments{
\item{in_channels}{Integer. Mel channels (2 = stereo).}

\item{base_channels,num_res_blocks,latent_channels,ch_mult,causality_axis}{See \code{\link{ltx23_audio_decoder}}.}
}
\value{
Module whose forward(x) returns the encoded audio latent, a
  tensor downsampled along time and mel axes with the configured
  latent channel count.
}
\description{
Mel spectrogram [B, 2, T, 64] -> latent distribution moments
[B, 2 * latent_channels, ceil(T/4), 16]. Structure mirrors the
decoder: causal convs, parameterless pixel norms, ResNet stages with
stride-2 downsampling between levels (reference LTX2AudioEncoder).
}
