% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_vocoder}
\alias{ltx23_vocoder}
\title{LTX-2.3 vocoder stage}
\usage{
ltx23_vocoder(in_channels = 128L, hidden_channels = 1536L, out_channels = 2L,
              upsample_kernel_sizes = c(11L, 4L, 4L, 4L, 4L, 4L),
              upsample_factors = c(5L, 2L, 2L, 2L, 2L, 2L),
              resnet_kernel_sizes = c(3L, 7L, 11L),
              resnet_dilations = list(c(1L, 3L, 5L), c(1L, 3L, 5L), c(1L, 3L, 5L)),
              antialias_ratio = 2L, antialias_kernel_size = 12L,
              final_bias = FALSE)
}
\arguments{
\item{in_channels}{Integer. Flattened input channels (C * mel bins / 1).}

\item{hidden_channels}{Integer.}

\item{out_channels}{Integer.}

\item{resnet_kernel_sizes}{Integer vector.}

\item{resnet_dilations}{List of integer vectors.}

\item{final_bias}{Logical.}

\item{upsample_kernel_sizes,upsample_factors}{Integer vectors.}

\item{antialias_ratio,antialias_kernel_size}{Integers.}
}
\description{
Mel spectrogram [B, C, T, M] -> waveform [B, out_channels, samples].
Channel and mel dims are flattened into conv channels; each upsample
stage halves the channel count and averages three parallel ResNet
branches.
}
