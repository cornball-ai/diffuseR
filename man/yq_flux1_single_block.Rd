% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_flux1_single_block}
\alias{yq_flux1_single_block}
\title{FLUX.1 single-stream block (anvl port)}
\usage{
yq_flux1_single_block(heads = 24L, head_dim = 128L, mlp_ratio = 4, eps = 1e-06,
                      precision = "highest")
}
\arguments{
\item{heads}{Integer. Attention heads.}

\item{head_dim}{Integer. Per-head dimension.}

\item{mlp_ratio}{Numeric. MLP hidden multiplier (4.0).}

\item{eps}{Numeric. Norm epsilon.}

\item{precision}{Character. Matmul precision.}
}
\value{
Function of (h, sil, cos, sin, w) returning the joint hidden
  states \code{[B, S, dim]}.
}
\description{
anvl re-implementation of \code{diffuseR::flux_single_block} +
\code{flux_attention(pre_only = TRUE)}: per-block adaLN-Zero
modulation (shift, scale, gate) over a parameterless LayerNorm,
\strong{non-fused} attention (\code{to_q/to_k/to_v}) and a parallel
GELU-tanh MLP (\code{proj_mlp}), joined by a single \code{proj_out}
over \code{cat(attn, mlp)} with a shared output gate.
}
