% tinyrox says don't edit this manually, but it can't stop you!
\name{flux2_parallel_self_attention}
\alias{flux2_parallel_self_attention}
\title{FLUX.2 parallel self-attention (single-stream)}
\usage{
flux2_parallel_self_attention(query_dim, heads, dim_head, mlp_ratio = 3,
                              eps = 1e-06, bias = FALSE)
}
\arguments{
\item{query_dim}{Integer. Model dimension.}

\item{heads}{Integer. Attention heads.}

\item{dim_head}{Integer. Per-head dimension.}

\item{mlp_ratio}{Numeric. MLP hidden multiplier (FLUX.2: 3.0).}

\item{eps}{Numeric. RMS norm epsilon.}

\item{bias}{Logical.}
}
\value{
Module whose forward(hidden_states, image_rotary_emb,
  chunk_size) returns the block output [B, S, query_dim]: attention
  and MLP branches computed in parallel from one fused projection,
  concatenated, and projected back by a second fused layer.
}
\description{
ViT-22B-style parallel block internals: one fused projection produces
QKV and the SwiGLU MLP input; one fused projection consumes
cat(attention output, MLP output). Reference:
Flux2ParallelSelfAttention + Flux2ParallelSelfAttnProcessor.
}
