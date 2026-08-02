% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_rms_norm}
\alias{ltx23_rms_norm}
\title{RMS normalization}
\usage{
ltx23_rms_norm(dim, eps = 1e-06, elementwise_affine = TRUE)
}
\arguments{
\item{dim}{Integer. Normalized dimension size.}

\item{eps}{Numeric. Stability epsilon.}

\item{elementwise_affine}{Logical. Learn a scale weight.}
}
\value{
Module whose forward(x) returns \code{x} RMS-normalized over
  the last axis and cast back to the input dtype, a tensor of the
  same shape.
}
\description{
Variance is computed in float32; the result is cast back to the input
dtype (or the weight dtype when elementwise affine).
}
