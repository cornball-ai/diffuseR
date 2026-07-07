% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_audio_resnet_block}
\alias{ltx23_audio_resnet_block}
\title{LTX audio ResNet block}
\usage{
ltx23_audio_resnet_block(in_channels, out_channels = NULL,
                         causality_axis = "height")
}
\arguments{
\item{causality_axis}{Character.}

\item{in_channels,out_channels}{Integers.}
}
\description{
PixelNorm -> SiLU -> causal conv, twice, with a 1x1 causal conv
shortcut (\code{nin_shortcut}) on channel change.
}
