% tinyrox says don't edit this manually, but it can't stop you!
\name{flux_double_block}
\alias{flux_double_block}
\title{FLUX double-stream (MMDiT) transformer block}
\usage{
flux_double_block(dim, num_attention_heads, attention_head_dim)
}
\arguments{
\item{dim}{Integer. Model dimension.}

\item{num_attention_heads}{Integer. Attention heads.}

\item{attention_head_dim}{Integer. Per-head dimension.}
}
\value{
Module whose forward(hidden_states, encoder_hidden_states,
  temb, image_rotary_emb) returns
  \code{list(encoder_hidden_states, hidden_states)}.
}
\description{
Image and text streams each get adaLN-Zero modulation and a
feed-forward; attention is joint across both streams. Reference:
diffusers FluxTransformerBlock.
}
