% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_latent_upsampler}
\alias{ltx23_latent_upsampler}
\title{LTX-2.3 latent upsampler model}
\usage{
ltx23_latent_upsampler(in_channels = 128L, mid_channels = 1024L,
                       num_blocks_per_stage = 4L)
}
\arguments{
\item{in_channels}{Integer. Latent channels.}

\item{mid_channels}{Integer.}

\item{num_blocks_per_stage}{Integer.}
}
\value{
Module whose forward(hidden_states) returns the 2x
  spatially upscaled latent, a tensor with the same batch, channel
  and frame counts and doubled height and width.
}
\description{
Latents [B, 128, F, H, W] -> [B, 128, F, 2H, 2W].
}
