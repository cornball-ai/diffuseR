% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_load_transformer_int8}
\alias{ltx23_load_transformer_int8}
\title{Load the LTX-2.3 transformer from an int8 artifact}
\usage{
ltx23_load_transformer_int8(ckpt, device = "cuda", pin = TRUE, verbose = TRUE,
                            ...)
}
\arguments{
\item{ckpt}{An \code{ltx23_checkpoint} from
\code{\link{ltx23_open_fp8_checkpoint}} on an int8 artifact.}

\item{device}{"cuda" or "cpu".}

\item{pin}{Logical. Pin the int8 host memory for full-rate DMA.}

\item{verbose}{Logical.}

\item{...}{Passed to \code{ltx23_transformer}.}
}
\value{
The transformer module.
}
\description{
Cast-set linears become \code{\link{ltx23_int8_linear}} modules
(int8 host-resident, optionally pinned, streamed per forward);
everything else loads as bfloat16 on \code{device}. Hard-errors on
unmapped or unfilled parameters.
}
