% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_downsample1d}
\alias{ltx23_downsample1d}
\title{Anti-aliasing 1D downsampler (low-pass then stride)}
\usage{
ltx23_downsample1d(ratio = 2L, kernel_size = NULL)
}
\arguments{
\item{ratio}{Integer. Downsampling ratio.}

\item{kernel_size}{Integer or NULL (default 6*ratio rounded even).}
}
\value{
Module whose forward(x) returns \code{x} low-pass filtered
  and decimated by \code{ratio} along the time axis.
}
\description{
Anti-aliasing 1D downsampler (low-pass then stride)
}
