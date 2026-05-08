% tinyrox says don't edit this manually, but it can't stop you!
\name{spatial_rational_resampler}
\alias{spatial_rational_resampler}
\title{Spatial Rational Resampler}
\usage{
spatial_rational_resampler(mid_channels, scale = 2)
}
\arguments{
\item{mid_channels}{Integer. Number of intermediate channels.}

\item{scale}{Numeric. Spatial scale factor (default 2.0).}
}
\value{
An \code{nn_module}.
}
\description{
Per-frame spatial upsampling: Conv2d -> PixelShuffle -> optional BlurDownsample.
For scale=2.0: num=2, den=1 (no blur downsampling needed).
}
\keyword{internal}
