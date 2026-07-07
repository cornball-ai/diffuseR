% tinyrox says don't edit this manually, but it can't stop you!
\name{flux_attention}
\alias{flux_attention}
\title{FLUX joint attention}
\usage{
flux_attention(query_dim, heads, dim_head, added_kv = FALSE, pre_only = FALSE,
               eps = 1e-06, bias = TRUE)
}
\arguments{
\item{query_dim}{Integer. Model dimension.}

\item{heads}{Integer. Number of attention heads.}

\item{dim_head}{Integer. Per-head dimension.}

\item{added_kv}{Logical. Add text-stream projections (double blocks).}

\item{pre_only}{Logical. Skip the output projection (single blocks).}

\item{eps}{Numeric. RMS norm epsilon.}

\item{bias}{Logical. Bias on the linear projections (TRUE for FLUX.1,
FALSE for FLUX.2).}
}
\description{
Multi-head attention with per-head RMS q/k norms and rotary position
embeddings. With \code{added_kv = TRUE} (double-stream blocks) the
text stream gets its own q/k/v projections and both streams attend
jointly (text tokens first); the outputs are split back and projected
per stream. With \code{pre_only = TRUE} (single-stream blocks) there
is no output projection. Reference: diffusers FluxAttention +
FluxAttnProcessor.
}
