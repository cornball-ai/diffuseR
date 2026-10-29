% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_load_transformer_nf4}
\alias{ltx23_load_transformer_nf4}
\title{Load the LTX-2.3 transformer with resident NF4 weights}
\usage{
ltx23_load_transformer_nf4(
  ckpt,
  device = "cuda",
  verbose = TRUE,
  lazy = FALSE,
  ...
)
}
\arguments{
\item{ckpt}{An NF4 \code{ltx23_checkpoint}
(\code{\link{ltx23_open_fp8_checkpoint}} reads any shard artifact).}

\item{device}{Character.}

\item{verbose}{Logical.}

\item{lazy}{Logical. Build the model and record each tensor's
artifact key without reading any weight; every weight is a
zero-stride placeholder until a disk-backed staging set reads it
onto the device. Host RAM then never holds the transformer.}

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
