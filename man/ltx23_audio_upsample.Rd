% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_audio_upsample}
\alias{ltx23_audio_upsample}
\title{LTX audio upsampler}
\usage{
ltx23_audio_upsample(in_channels, causality_axis = "height")
}
\arguments{
\item{in_channels}{Integer.}

\item{causality_axis}{Character.}
}
\value{
Module whose forward(x) returns the tensor upsampled 2x by
  nearest-neighbour interpolation and convolved.
}
\description{
Nearest 2x interpolation, causal conv, then a crop of the first
element along the causal axis.
}
