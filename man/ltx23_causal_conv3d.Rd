% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_causal_conv3d}
\alias{ltx23_causal_conv3d}
\title{Causal 3D convolution}
\usage{
ltx23_causal_conv3d(in_channels, out_channels, kernel_size = 3L, stride = 1L,
                    spatial_padding_mode = "zeros")
}
\arguments{
\item{kernel_size}{Integer or length-3 vector (t, h, w).}

\item{stride}{Integer or length-3 vector.}

\item{spatial_padding_mode}{Character. Conv padding mode.}

\item{in_channels,out_channels}{Integers.}
}
\value{
Module whose forward(hidden_states, causal) returns the 3-D
  convolution of the input. With \code{causal = TRUE} the temporal
  axis is left-padded by replicating the first frame, so no output
  frame sees a later input frame.
}
\description{
Spatial padding is handled by the convolution; temporal padding
replicates the first frame (causal) or both edge frames (non-causal),
chosen at call time.
}
