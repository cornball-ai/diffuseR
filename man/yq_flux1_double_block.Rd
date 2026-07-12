% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_flux1_double_block}
\alias{yq_flux1_double_block}
\title{FLUX.1 double-stream (MMDiT) block (anvl port)}
\usage{
yq_flux1_double_block(heads = 24L, head_dim = 128L, eps = 1e-06,
                      precision = "highest")
}
\arguments{
\item{heads}{Integer. Attention heads.}

\item{head_dim}{Integer. Per-head dimension.}

\item{eps}{Numeric. Norm epsilon.}

\item{precision}{Character. Matmul precision (see
\code{\link[yunque]{yq_linear}}).}
}
\value{
Function of (h, c, sil, cos, sin, w) returning
  \code{list(c_out, h_out)}.
}
\description{
anvl re-implementation of \code{diffuseR::flux_double_block} +
\code{flux_attention(added_kv = TRUE)}: separate image and text
streams, each with per-block adaLN-Zero modulation (computed inside
the block from \code{silu(temb)}) and a GELU-tanh feed-forward, joined
by a single attention over the concatenated [text; image] sequence
(text tokens first, matching the rotary layout).
}
\details{
Returns a closure over the static config; \code{anvl::jit()} the
enclosing transformer. Weights travel as a named list mirroring the
checkpoint keys under \code{transformer_blocks.N.}.

}
