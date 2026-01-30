% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx2_video_causal_conv3d}
\alias{ltx2_video_causal_conv3d}
\title{LTX2 Video Causal 3D Convolution}
\usage{
ltx2_video_causal_conv3d(
  in_channels,
  out_channels,
  kernel_size = 3L,
  stride = 1L,
  dilation = 1L,
  groups = 1L,
  spatial_padding_mode = "zeros"
)
}
\arguments{
\item{in_channels}{Integer. Input channels.}

\item{out_channels}{Integer. Output channels.}

\item{kernel_size}{Integer or vector of 3. Convolution kernel size.}

\item{stride}{Integer or vector of 3. Stride.}

\item{dilation}{Integer or vector of 3. Dilation.}

\item{groups}{Integer. Convolution groups. Default: 1}

\item{spatial_padding_mode}{Character. Padding mode for spatial dims. Default: "zeros"}
}
\description{
3D convolution with runtime-selectable causal or non-causal padding.
Causal mode pads temporally by repeating first frame.
Non-causal mode pads temporally by repeating first and last frames.
}
