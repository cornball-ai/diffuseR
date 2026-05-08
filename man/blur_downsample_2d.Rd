% tinyrox says don't edit this manually, but it can't stop you!
\name{blur_downsample_2d}
\alias{blur_downsample_2d}
\title{BlurDownsample (anti-aliased spatial downsampling)}
\usage{
blur_downsample_2d(stride, kernel_size = 5L)
}
\arguments{
\item{stride}{Integer. Downsampling stride.}

\item{kernel_size}{Integer. Blur kernel size (default 5).}
}
\value{
An \code{nn_module}.
}
\description{
Fixed separable binomial kernel for anti-aliased downsampling.
With stride=1 this is the identity.
}
\keyword{internal}
