% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_video_down_block3d}
\alias{ltx23_video_down_block3d}
\title{LTX video down block}
\usage{
ltx23_video_down_block3d(in_channels, out_channels = NULL, num_layers = 1L,
                         resnet_eps = 1e-06, spatio_temporal_scale = TRUE,
                         downsample_type = "spatiotemporal",
                         spatial_padding_mode = "zeros")
}
\arguments{
\item{num_layers}{Integer. ResNet count.}

\item{resnet_eps}{Numeric.}

\item{spatio_temporal_scale}{Logical. Whether to downsample at all.}

\item{downsample_type}{"spatial", "temporal", or "spatiotemporal".}

\item{spatial_padding_mode}{Character.}

\item{in_channels,out_channels}{Integers.}
}
\value{
Module whose forward(hidden_states, causal) returns the
  stage output: the resnet stack applied in sequence, then the
  optional downsampler.
}
\description{
ResNet stack (at the input channel count) followed by a
pixel-unshuffle downsampler that also changes the channel count.
}
