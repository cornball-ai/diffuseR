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
\description{
Upsample 2x, apply the activation, downsample 2x.
}
