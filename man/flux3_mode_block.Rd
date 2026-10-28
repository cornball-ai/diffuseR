% tinyrox says don't edit this manually, but it can't stop you!
\name{flux3_mode_block}
\alias{flux3_mode_block}
\title{FLUX 3 per-mode transformer block}
\usage{
flux3_mode_block(
  dim,
  num_attention_heads,
  attention_head_dim,
  mlp_ratio = 3,
  eps = 1e-06
)
}
\arguments{
\item{dim}{Integer. Model dimension.}

\item{num_attention_heads}{Integer. Attention heads.}

\item{attention_head_dim}{Integer. Per-head dimension.}

\item{mlp_ratio}{Numeric. Feed-forward multiplier (3.0).}

\item{eps}{Numeric. Norm epsilon.}
}
\value{
Module whose forward(hidden_states, temb_mod,
  image_rotary_emb, chunk_size) returns the updated states.
}
\description{
Parameterless LayerNorm, per-token (shift, scale) modulation, then
parallel attention and feed-forward from the same normed input, added
back under one gate. Self-attention stays within a single token mode.
Reference: Flux3ModeTransformerBlock.
}
