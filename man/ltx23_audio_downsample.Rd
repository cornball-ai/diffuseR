% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_audio_downsample}
\alias{ltx23_audio_downsample}
\title{LTX audio downsampler}
\usage{
ltx23_audio_downsample(in_channels, causality_axis = "height")
}
\arguments{
\item{in_channels}{Integer.}

\item{causality_axis}{Character.}
}
\value{
Module whose forward(x) returns the strided convolution of
  \code{x}, halving the downsampled axes.
}
\description{
Causal zero-pad followed by a plain stride-2 conv (reference
LTX2AudioDownsample; note the conv is unwrapped, so its checkpoint
key is \code{downsample.conv.*}).
}
