% tinyrox says don't edit this manually, but it can't stop you!
\name{flux3_single_block}
\alias{flux3_single_block}
\title{FLUX 3 joint (single-stream) block}
\usage{
flux3_single_block(
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
Module whose forward(hidden_states, encoder_hidden_states,
  temb_mod, temb_mod_context, image_rotary_emb, chunk_size) returns
  \code{list(encoder_hidden_states, hidden_states)}.
}
\description{
Text and content tokens are normed and modulated separately, then
concatenated (text first) for one parallel attention + feed-forward;
the output is split back and added under per-stream gates.
Reference: Flux3SingleTransformerBlock.
}
