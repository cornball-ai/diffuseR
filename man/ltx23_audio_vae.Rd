% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_audio_vae}
\alias{ltx23_audio_vae}
\title{LTX-2.3 audio VAE}
\usage{
ltx23_audio_vae(base_channels = 128L, output_channels = 2L,
                num_res_blocks = 2L, latent_channels = 8L,
                ch_mult = c(1L, 2L, 4L), causality_axis = "height",
                mel_bins = 64L)
}
\arguments{
\item{base_channels,output_channels,num_res_blocks,latent_channels,ch_mult,causality_axis,mel_bins}{See \code{\link{ltx23_audio_decoder}}.}
}
\description{
Decoder plus the per-channel latent statistics loaded from the
checkpoint. Only decoding is supported.
}
