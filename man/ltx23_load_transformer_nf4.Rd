% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_load_transformer_nf4}
\alias{ltx23_load_transformer_nf4}
\title{Load the LTX-2.3 transformer with resident NF4 weights}
\usage{
ltx23_load_transformer_nf4(ckpt, device = "cuda", verbose = TRUE, ...)
}
\arguments{
\item{ckpt}{An NF4 \code{ltx23_checkpoint}
(\code{\link{ltx23_open_fp8_checkpoint}} reads any shard artifact).}

\item{device}{Character.}

\item{verbose}{Logical.}

\item{...}{Passed to \code{\link{ltx23_transformer}} (tiny test configs).}
}
\value{
The loaded \code{ltx23_transformer}.
}
\description{
Builds the transformer, swaps the cast-set linears for
\code{\link{ltx23_nf4_linear}}, and loads everything onto
\code{device}: at ~4.5 bits/parameter the whole 22B transformer stays
GPU-resident, avoiding per-step weight transfers.
}
