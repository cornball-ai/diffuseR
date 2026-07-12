% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_gemma3_encoder}
\alias{yq_gemma3_encoder}
\title{Gemma3 text encoder forward for LTX-2.3 (anvl port)}
\usage{
yq_gemma3_encoder(num_layers = 6L, num_heads = 4L, num_kv = 2L, head_dim = 48L,
                  eps = 1e-06, sliding_window_pattern = 6L,
                  precision = "highest")
}
\arguments{
\item{num_layers}{Integer. Decoder layers.}

\item{eps}{Numeric. RMS norm epsilon.}

\item{sliding_window_pattern}{Integer. Global-attention period (6).}

\item{precision}{Character. Matmul precision.}

\item{num_heads,num_kv,head_dim}{Integers. Attention shape.}
}
\value{
Function of (embeds, rope_global, rope_local, mask, w):
  \itemize{
    \item embeds \code{[B, S, hidden]} from \code{\link{yq_gemma3_embed}}
    \item rope_global, rope_local \code{list(cos, sin)} \code{[S, D/2]}
      from \code{\link{yq_gemma3_rope}}
    \item mask \code{[B, 1, S, S]} from \code{\link{yq_gemma3_mask}}
    \item w weights pytree from \code{\link{yq_gemma3_load_weights}}
  }
  returning \code{[B, S, hidden, num_layers + 1]}.
}
\description{
anvl re-implementation of \code{diffuseR::gemma3_text_model}, returning
the stacked hidden states the LTX connectors consume: the scaled
embedding output plus every layer's output, with the final layer's
output final-RMS-normed (\code{num_layers + 1} states). Mirrors the HF
hidden-state semantics: state \code{i} is recorded BEFORE layer \code{i}
and the last state is post-final-norm (the un-normed last-layer output
never appears). Per-layer the dual RoPE table is selected by the same
rule as the reference — the every-\code{pattern}-th layer (global) uses
the global table, the rest (sliding) use the local table — and one full
causal+padding mask is applied to every layer.
}
