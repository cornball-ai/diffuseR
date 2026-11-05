% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_latent_upsampler}
\alias{ltx23_latent_upsampler}
\title{LTX latent upsampler model}
\usage{
ltx23_latent_upsampler(
  in_channels = 128L,
  mid_channels = 1024L,
  num_blocks_per_stage = 4L,
  temporal = FALSE
)
}
\arguments{
\item{in_channels}{Integer. Latent channels.}

\item{mid_channels}{Integer (1024 for the spatial upscalers, 512 for
the LTX-2.5 temporal one).}

\item{num_blocks_per_stage}{Integer.}

\item{temporal}{Logical. Build the temporal x2 layout.}
}
\value{
Module whose forward(hidden_states) returns the upscaled
  latent.
}
\description{
Spatial: latents [B, 128, F, H, W] -> [B, 128, F, 2H, 2W] (the LTX-2.3
and LTX-2.5 spatial x2 upscalers). Temporal: [B, 128, F, H, W] ->
[B, 128, 2F - 1, H, W] (the LTX-2.5 temporal x2 upscaler; the first
of the doubled frames is dropped, so 8k + 1 pixel frames become
16k + 1).
}
