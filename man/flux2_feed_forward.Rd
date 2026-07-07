% tinyrox says don't edit this manually, but it can't stop you!
\name{flux2_feed_forward}
\alias{flux2_feed_forward}
\title{FLUX.2 feed-forward (fused SwiGLU)}
\usage{
flux2_feed_forward(dim, dim_out = NULL, mult = 3, bias = FALSE)
}
\arguments{
\item{dim}{Integer. Input dimension.}

\item{dim_out}{Integer. Output dimension (defaults to \code{dim}).}

\item{mult}{Numeric. Inner dim multiplier (FLUX.2: 3.0).}

\item{bias}{Logical.}
}
\description{
\code{linear_in} projects to twice the inner dim; SwiGLU gates the
first half with SiLU and multiplies by the second half;
\code{linear_out} projects back. Reference: Flux2FeedForward +
Flux2SwiGLU.
}
