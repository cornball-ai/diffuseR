% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_vocoder_resblock}
\alias{ltx23_vocoder_resblock}
\title{Vocoder ResNet block (AMP)}
\usage{
ltx23_vocoder_resblock(channels, kernel_size = 3L, dilations = c(1L, 3L, 5L),
                       antialias_ratio = 2L, antialias_kernel_size = 12L)
}
\arguments{
\item{channels}{Integer.}

\item{kernel_size}{Integer.}

\item{dilations}{Integer vector.}

\item{antialias_ratio,antialias_kernel_size}{Integers.}
}
\value{
Module whose forward(x) returns \code{x} after the dilated
  convolution pairs have been added back as residuals, a tensor of
  the same shape.
}
\description{
Dilated conv pairs, each preceded by an anti-aliased SnakeBeta
activation, with residual connections.
}
