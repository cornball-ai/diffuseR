% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_connector_transformer_1d}
\alias{ltx23_connector_transformer_1d}
\title{1D connector transformer}
\usage{
ltx23_connector_transformer_1d(num_attention_heads = 32L,
                               attention_head_dim = 128L, num_layers = 8L,
                               num_learnable_registers = 128L,
                               rope_base_seq_len = 4096L, rope_theta = 10000,
                               rope_double_precision = TRUE, eps = 1e-06,
                               rope_type = "split", gated_attention = TRUE)
}
\arguments{
\item{num_learnable_registers}{Integer or NULL. Register count (the
sequence length must be divisible by it).}

\item{eps}{Numeric. Norm epsilon.}

\item{gated_attention}{Logical. Per-head attention output gates.}

\item{num_attention_heads,attention_head_dim,num_layers}{Transformer shape.}

\item{rope_base_seq_len,rope_theta,rope_double_precision,rope_type}{RoPE config.}
}
\value{
Module whose forward(hidden_states, attention_mask,
  attn_mask_binarize_threshold) returns
  \code{list(hidden_states, attention_mask)}: the transformed
  sequence and the (possibly binarized) mask that accompanies it.
}
\description{
Replaces padded positions with learnable registers (valid tokens are
front-aligned in their original order; the tail is filled with
registers indexed by absolute position, after which the attention mask
is cleared), then runs 1D transformer blocks with rotary embeddings.
}
