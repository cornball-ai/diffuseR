% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_antialias_act1d}
\alias{ltx23_antialias_act1d}
\title{Anti-aliased activation}
\usage{
ltx23_antialias_act1d(channels, ratio = 2L, kernel_size = 12L)
}
\arguments{
\item{channels}{Integer. Channels for the SnakeBeta activation.}

\item{ratio,kernel_size}{Integers. Resampling config.}
}
\value{
Module whose forward(x) returns the activation applied at 2x
  rate (upsample, activate, downsample), a tensor of the same shape
  as \code{x}, with the aliasing the raw activation would introduce
  filtered out.
}
\description{
Upsample 2x, apply the activation, downsample 2x.
}
