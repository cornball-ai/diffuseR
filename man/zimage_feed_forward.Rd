% tinyrox says don't edit this manually, but it can't stop you!
\name{zimage_feed_forward}
\alias{zimage_feed_forward}
\title{Z-Image feed-forward (SwiGLU with separate gate weights)}
\usage{
zimage_feed_forward(dim, hidden_dim)
}
\arguments{
\item{dim}{Integer. Model width.}

\item{hidden_dim}{Integer. Hidden width.}
}
\value{
Module whose forward(x) returns \code{w2(silu(w1(x)) * w3(x))},
  a tensor of the same shape as \code{x}.
}
\description{
w2(silu(w1(x)) * w3(x)) with all three linears bias-free. The hidden
width is int(dim / 3 * 8).
}
