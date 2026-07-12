% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_t5_encoder}
\alias{yq_t5_encoder}
\title{T5-XXL encoder forward (jit-ready closure)}
\usage{
yq_t5_encoder(num_layers = 24L, num_heads = 64L, d_kv = 64L, eps = 1e-06,
              precision = "highest")
}
\arguments{
\item{num_layers}{Integer. Encoder blocks (XXL: 24).}

\item{eps}{Numeric. RMS norm epsilon.}

\item{precision}{Character. Matmul precision.}

\item{num_heads,d_kv}{Integers. Attention shape (XXL: 64 heads x 64).}
}
\value{
Function of \code{(embeds, position_bias, w)}:
  \itemize{
    \item embeds \code{[B, S, d_model]} from \code{\link{yq_t5_embed}}
    \item position_bias \code{[1, H, S, S]} from
      \code{\link{yq_t5_rel_pos_bias}}
    \item w weights pytree from \code{\link{yq_t5_load_weights}}
  }
  returning \code{[B, S, d_model]}.
}
\description{
Runs \code{num_layers} T5 encoder blocks over pre-embedded tokens and
applies the final RMS norm, returning the last hidden state. Defaults
are the T5-v1.1-XXL configuration used by FLUX.1.
}
