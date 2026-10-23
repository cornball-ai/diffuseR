% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_feed_forward}
\alias{ltx23_feed_forward}
\title{LTX feed-forward layer}
\usage{
ltx23_feed_forward(dim, mult = 4L, bias = TRUE)
}
\arguments{
\item{dim}{Integer. Input/output dimension.}

\item{mult}{Integer. Hidden dimension multiplier.}

\item{bias}{Logical. Both linears carry a bias (FALSE for the LTX-2.5
video stream).}
}
\value{
Module whose forward(x) returns the projected states passed
  through a tanh-approximated GELU, a tensor of the same shape as
  \code{x}.
}
\description{
Linear -> GELU (tanh approximation) -> Linear with 4x hidden dim,
matching diffusers \code{FeedForward(activation_fn="gelu-approximate")}
state-dict names (net.0.proj, net.2).
}
