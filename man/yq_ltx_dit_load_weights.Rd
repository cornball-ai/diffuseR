% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_ltx_dit_load_weights}
\alias{yq_ltx_dit_load_weights}
\title{Load LTX-2.3 DiT weights into an anvl pytree}
\usage{
yq_ltx_dit_load_weights(path, isolate = TRUE, device = "cpu")
}
\arguments{
\item{path}{Path to the state-dict safetensors file.}

\item{isolate}{Logical. If \code{FALSE}, also load the a2v/v2a
cross-attention and global av-modulation weights.}

\item{device}{Character. Target device.}
}
\value{
Weights pytree for \code{\link{yq_ltx_dit}}.
}
\description{
Reads a \code{ltx23_transformer} state-dict safetensors file (f32),
transposing 2-D linears to \code{[in, out]} and wrapping each tensor as
an \code{AnvlArray} on \code{device}. Modulation tables (scale/shift)
load in their raw \code{[num_params, dim]} shape. Block count is derived
from the \code{transformer_blocks.N} key census.
}
