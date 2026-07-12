% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_sdxl_bigg_encoder}
\alias{yq_sdxl_bigg_encoder}
\title{SDXL OpenCLIP bigG penultimate + pooled encoder forward (anvl)}
\usage{
yq_sdxl_bigg_encoder(num_layers = 32L, num_heads = 20L, head_dim = 64L,
                     eps = 1e-05, precision = "highest")
}
\arguments{
\item{num_layers}{Integer. Transformer layers (bigG: 32).}

\item{num_heads}{Integer. Attention heads (20).}

\item{head_dim}{Integer. Per-head dimension (64).}

\item{eps}{Numeric. LayerNorm epsilon.}

\item{precision}{Character. Matmul precision.}
}
\value{
Function of \code{(embeds, mask, w)} returning a list:
  \itemize{
    \item \code{penult} \code{[B, S, 1280]} penultimate hidden state
    \item \code{hidden_ln} \code{[B, S, 1280]} final-LN full hidden state
  }
}
\description{
Returns a closure over the static bigG configuration;
\code{anvl::jit()} the closure. Runs all \code{num_layers} transformer
layers over pre-embedded tokens (exact/erf GELU MLP), capturing the
penultimate hidden state (after \code{num_layers - 1} layers,
\code{hidden_states[-2]}) and applying the final LayerNorm to the full
output. Returns both so the caller can gather the pooled EOS row and
project it (see \code{\link{yq_sdxl_clip_encoders}}).
}
