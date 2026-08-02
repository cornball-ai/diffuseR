% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_video_resnet_block3d}
\alias{ltx23_video_resnet_block3d}
\title{LTX 3D ResNet block}
\usage{
ltx23_video_resnet_block3d(in_channels, out_channels = NULL, eps = 1e-06,
                           spatial_padding_mode = "zeros")
}
\arguments{
\item{eps}{Numeric. Shortcut LayerNorm epsilon.}

\item{spatial_padding_mode}{Character.}

\item{in_channels,out_channels}{Integers.}
}
\value{
Module whose forward(inputs, causal) returns \code{inputs}
  plus the residual branch, a tensor of the same shape.
}
\description{
PerChannelRMSNorm -> SiLU -> causal conv, twice, with a LayerNorm +
1x1 Conv3d shortcut when the channel count changes.
}
