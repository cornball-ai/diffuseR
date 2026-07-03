% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_audio_causal_conv2d}
\alias{ltx23_audio_causal_conv2d}
\title{Causal 2D convolution for audio spectrograms}
\usage{
ltx23_audio_causal_conv2d(in_channels, out_channels, kernel_size = 3L,
                          stride = 1L, causality_axis = "height")
}
\arguments{
\item{kernel_size}{Integer or length-2 vector.}

\item{stride}{Integer.}

\item{causality_axis}{"height", "width", "width-compatibility", or "none".}

\item{in_channels,out_channels}{Integers.}
}
\description{
Pads asymmetrically along the causal axis ("height" = time frames for
LTX audio) before an unpadded Conv2d.
}
