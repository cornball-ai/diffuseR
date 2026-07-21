% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_int8_quantize}
\alias{ltx23_int8_quantize}
\title{Quantize a tensor to int8 with per-output-channel scales}
\usage{
ltx23_int8_quantize(x)
}
\arguments{
\item{x}{Float 2D weight [out, in].}
}
\value{
List with \code{weight_int8} (int8 [out, in]) and
  \code{scale} (float32 [out], absmax/127 per row).
}
\description{
Quantize a tensor to int8 with per-output-channel scales
}
