% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx_video_upsampler3d}
\alias{ltx_video_upsampler3d}
\title{LTX Video Upsampler 3D}
\usage{
ltx_video_upsampler3d(
  in_channels,
  stride = 1L,
  residual = FALSE,
  upscale_factor = 1L,
  spatial_padding_mode = "zeros"
)
}
\arguments{
\item{in_channels}{Integer. Input channels.}

\item{stride}{Integer or vector of 3. Upsampling stride.}

\item{residual}{Logical. Whether to use residual connection.}

\item{upscale_factor}{Integer. Channel upscale factor.}

\item{spatial_padding_mode}{Character. Padding mode.}
}
\description{
Spatiotemporal upsampling with pixel shuffle + optional residual.
}
