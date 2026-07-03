% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_load_transformer_fp8}
\alias{ltx23_load_transformer_fp8}
\title{Load the LTX-2.3 transformer with FP8 weights}
\usage{
ltx23_load_transformer_fp8(ckpt, device = "cuda", pin = TRUE, verbose = TRUE,
                           ...)
}
\arguments{
\item{ckpt}{An fp8 \code{ltx23_checkpoint}
(\code{\link{ltx23_open_fp8_checkpoint}}).}

\item{device}{Character. Device for the resident (non-fp8) weights.}

\item{pin}{Logical. Pin the fp8 host memory for faster transfers.}

\item{verbose}{Logical.}

\item{...}{Passed to \code{\link{ltx23_transformer}} (tiny test configs).}
}
\value{
The loaded \code{ltx23_transformer}.
}
\description{
Builds the transformer, swaps the official cast-set linears for
\code{\link{ltx23_fp8_linear}}, loads fp8 weights CPU-side (optionally
pinned) and everything else as bfloat16 on \code{device}. Sets
\code{options(diffuseR.use_fp8 = TRUE)} so the transformer runs
per-block garbage collection over the dequantized temporaries.
}
