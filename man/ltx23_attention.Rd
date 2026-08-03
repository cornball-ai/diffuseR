% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_attention}
\alias{ltx23_attention}
\title{LTX-2 attention layer}
\usage{
ltx23_attention(query_dim, heads = 8L, kv_heads = NULL, dim_head = 64L,
                bias = TRUE, cross_attention_dim = NULL, out_bias = TRUE,
                norm_eps = 1e-06, norm_elementwise_affine = TRUE,
                rope_type = "split", apply_gated_attention = FALSE)
}
\arguments{
\item{query_dim}{Integer. Query feature dimension.}

\item{dim_head}{Integer. Per-head dimension.}

\item{cross_attention_dim}{Integer or NULL. Key/value input dimension
(NULL for self-attention).}

\item{norm_eps}{Numeric. RMS norm epsilon.}

\item{norm_elementwise_affine}{Logical. RMS norms carry weights.}

\item{rope_type}{"split" or "interleaved".}

\item{apply_gated_attention}{Logical. Add per-head sigmoid output gates.}

\item{heads,kv_heads}{Integers. Attention head counts.}

\item{bias,out_bias}{Logicals. Projection biases.}
}
\value{
Module whose forward(hidden_states, ...) returns the attended
  states [B, S, query_dim] after the output projection.
}
\description{
Attention with RMS q/k norms across heads, optional per-head output
gating (LTX-2.3), separate query/key RoPE (for a2v/v2a cross
attention), and optional STG perturbation (skip attention, use the
value projection).
}
