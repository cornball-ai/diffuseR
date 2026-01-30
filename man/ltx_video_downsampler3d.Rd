% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx_video_downsampler3d}
\alias{ltx_video_downsampler3d}
\title{LTX Video Downsampler 3D}
\usage{
ltx_video_downsampler3d(
  in_channels,
  out_channels,
  stride = 1L,
  spatial_padding_mode = "zeros"
)
}
\arguments{
\item{in_channels}{Integer. Input channels.}

\item{out_channels}{Integer. Output channels.}

\item{stride}{Integer or vector of 3. Downsampling stride.}

\item{spatial_padding_mode}{Character. Padding mode.}
}
\description{
Spatiotemporal downsampling with strided pixel unshuffle + convolution.
}
