% tinyrox says don't edit this manually, but it can't stop you!
\name{flux3_attention}
\alias{flux3_attention}
\title{FLUX 3 self-attention}
\usage{
flux3_attention(query_dim, heads, dim_head, eps = 1e-06)
}
\arguments{
\item{query_dim}{Integer. Model dimension.}

\item{heads}{Integer. Attention heads.}

\item{dim_head}{Integer. Per-head dimension.}

\item{eps}{Numeric. RMS norm epsilon.}
}
\value{
Module whose forward(hidden_states, image_rotary_emb,
  chunk_size) returns the attended states [B, S, query_dim].
}
\description{
Separate q/k/v projections, per-head RMS norms and rotary embeddings
applied in float32 (cast back to the value dtype before attention),
and one output projection. Reference: Flux3Attention +
Flux3AttnProcessor.
}
