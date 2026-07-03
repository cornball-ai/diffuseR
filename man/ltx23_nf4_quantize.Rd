% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_nf4_quantize}
\alias{ltx23_nf4_quantize}
\title{Quantize a tensor to NF4}
\usage{
ltx23_nf4_quantize(x)
}
\arguments{
\item{x}{Float tensor (any shape; total elements must be a multiple
of 128, i.e. two 64-element blocks - always true for the LTX
linears).}
}
\value{
List with \code{packed} (uint8, two indices per byte) and
  \code{absmax} (float32, one per 64-element block).
}
\description{
Quantize a tensor to NF4
}
