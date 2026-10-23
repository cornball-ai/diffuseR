% tinyrox says don't edit this manually, but it can't stop you!
\name{gemma4_rms_norm}
\alias{gemma4_rms_norm}
\title{Gemma 4 RMS normalization}
\usage{
gemma4_rms_norm(dim, eps = 1e-06, with_scale = TRUE)
}
\arguments{
\item{dim}{Integer. Normalized dimension.}

\item{eps}{Numeric. Epsilon.}

\item{with_scale}{Logical. Carry a learned scale (FALSE for the value
norm).}
}
\value{
Module whose forward(x) normalizes over the last dimension in
  float32 and multiplies by the scale, returning \code{x}'s dtype.
}
\description{
Gemma 4 RMS normalization
}
\keyword{internal}
