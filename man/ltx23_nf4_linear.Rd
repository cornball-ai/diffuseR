% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_nf4_linear}
\alias{ltx23_nf4_linear}
\title{NF4 linear layer}
\usage{
ltx23_nf4_linear(out_features, in_features, bias = TRUE)
}
\arguments{
\item{bias}{Logical.}

\item{out_features,in_features}{Integers.}
}
\value{
Module whose forward(x) returns the linear projection of
  \code{x}, dequantizing the NF4 weight into a reusable buffer first.
  Same result as an \code{nn_linear} of the same shape, at roughly an
  eighth of the resident weight bytes.
}
\description{
Packed weights and per-block scales are registered as buffers, so
they move with the module (uint8 packs are untouched by dtype
conversions). The forward pass dequantizes on the weight's device.
}
