% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_video_mid_block3d}
\alias{ltx23_video_mid_block3d}
\title{LTX video mid block}
\usage{
ltx23_video_mid_block3d(in_channels, num_layers = 1L, resnet_eps = 1e-06,
                        spatial_padding_mode = "zeros")
}
\arguments{
\item{in_channels}{Integer.}

\item{num_layers}{Integer.}

\item{resnet_eps}{Numeric.}

\item{spatial_padding_mode}{Character.}
}
\value{
Module whose forward(hidden_states, causal) returns the
  bottleneck output, a tensor of the same shape as the input.
}
\description{
A plain ResNet stack at a fixed channel count.
}
