% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_qwen3_encoder}
\alias{yq_qwen3_encoder}
\title{Qwen3 encoder forward for FLUX.2 (anvl port)}
\usage{
yq_qwen3_encoder(out_layers = c(9L, 18L, 27L), num_heads = 32L, num_kv = 8L,
                 head_dim = 128L, eps = 1e-06, precision = "highest")
}
\arguments{
\item{out_layers}{Integer vector. 1-based layer depths whose outputs
are concatenated (klein-4B: 9, 18, 27).}

\item{eps}{Numeric. RMS norm epsilon.}

\item{precision}{Character. Matmul precision.}

\item{num_heads,num_kv,head_dim}{Integers. Attention shape.}
}
\value{
Function of (embeds, cos, sin, mask, w):
  \itemize{
    \item embeds \code{[B, S, hidden]} from \code{\link{yq_qwen3_embed}}
    \item cos, sin \code{[S, head_dim/2]} from \code{\link{yq_qwen3_rope}}
    \item mask \code{[B, 1, S, S]} from \code{\link{yq_qwen3_mask}}
    \item w weights pytree (\code{w$layers[[i]]})
  }
  returning \code{[B, S, length(out_layers) * hidden]}.
}
\description{
anvl re-implementation of \code{diffuseR::qwen3_encoder}, restricted
to what FLUX.2 klein needs: runs to \code{max(out_layers)} decoder
layers over pre-embedded tokens and returns the requested mid-stack
hidden states concatenated per token (no final norm, no LM head).
Defaults are Qwen3-4B.
}
