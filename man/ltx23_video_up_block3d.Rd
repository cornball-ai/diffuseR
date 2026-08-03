% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_video_up_block3d}
\alias{ltx23_video_up_block3d}
\title{LTX video up block}
\usage{
ltx23_video_up_block3d(in_channels, out_channels = NULL, num_layers = 1L,
                       resnet_eps = 1e-06, spatio_temporal_scale = TRUE,
                       upsample_type = "spatiotemporal",
                       upsample_residual = FALSE, upscale_factor = 1L,
                       spatial_padding_mode = "zeros")
}
\arguments{
\item{num_layers}{Integer.}

\item{resnet_eps}{Numeric.}

\item{spatio_temporal_scale}{Logical.}

\item{upsample_type}{"spatial", "temporal", or "spatiotemporal".}

\item{upsample_residual}{Logical.}

\item{upscale_factor}{Integer.}

\item{spatial_padding_mode}{Character.}

\item{in_channels,out_channels}{Integers.}
}
\value{
Module whose forward(hidden_states, causal) returns the
  stage output: the optional input projection and upsampler, then the
  resnet stack applied in sequence.
}
\description{
Optional channel-changing conv-in ResNet, pixel-shuffle upsampler,
then a ResNet stack at the output channel count.
}
