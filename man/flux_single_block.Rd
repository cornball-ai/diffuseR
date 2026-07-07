% tinyrox says don't edit this manually, but it can't stop you!
\name{flux_single_block}
\alias{flux_single_block}
\title{FLUX single-stream transformer block}
\usage{
flux_single_block(dim, num_attention_heads, attention_head_dim, mlp_ratio = 4)
}
\arguments{
\item{dim}{Integer. Model dimension.}

\item{num_attention_heads}{Integer. Attention heads.}

\item{attention_head_dim}{Integer. Per-head dimension.}

\item{mlp_ratio}{Numeric. MLP hidden dim multiplier.}
}
\value{
Module whose forward(hidden_states, temb, image_rotary_emb)
  returns the joint hidden states.
}
\description{
Parallel attention + MLP over the joint [text; image] sequence with a
shared gate: \code{x + gate * proj_out(cat(attn, gelu(mlp)))}. The
reference concatenates the streams inside every block and splits after;
here the caller concatenates once before the single-block stack, which
is numerically identical. Reference: diffusers
FluxSingleTransformerBlock.
}
