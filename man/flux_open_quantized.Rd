% tinyrox says don't edit this manually, but it can't stop you!
\name{flux_open_quantized}
\alias{flux_open_quantized}
\title{Open a quantized FLUX artifact directory}
\usage{
flux_open_quantized(dir)
}
\arguments{
\item{dir}{The quantized artifact directory (with manifest.json).}
}
\value{
An \code{ltx23_checkpoint} with \code{$format} set.
}
\description{
Opens the sharded NF4/fp8 artifact written by
\code{\link{flux_quantize}} through the shared checkpoint interface.
The manifest's embedded transformer config and \code{format} ride
along, so \code{\link{flux_load_transformer}} needs nothing else.
}
