% tinyrox says don't edit this manually, but it can't stop you!
\name{flux2_single_block}
\alias{flux2_single_block}
\title{FLUX.2 single-stream block (parallel)}
\usage{
flux2_single_block(dim, num_attention_heads, attention_head_dim, mlp_ratio = 3,
                   eps = 1e-06, bias = FALSE)
}
\arguments{
\item{dim}{Integer. Model dimension.}

\item{num_attention_heads}{Integer. Attention heads.}

\item{attention_head_dim}{Integer. Per-head dimension.}

\item{mlp_ratio}{Numeric. MLP multiplier (FLUX.2: 3.0).}

\item{eps}{Numeric. Norm epsilon.}

\item{bias}{Logical.}
}
\value{
Module whose forward(hidden_states, temb_mod,
  image_rotary_emb) returns the joint hidden states.
}
\description{
Parameterless LayerNorm with external modulation, then the fused
parallel attention+MLP. Operates on the pre-concatenated [text; image]
sequence (the reference model concatenates once before the stack).
Reference: Flux2SingleTransformerBlock.
}
