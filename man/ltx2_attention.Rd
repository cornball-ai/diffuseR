% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx2_attention}
\alias{ltx2_attention}
\title{LTX2 Attention module}
\usage{
ltx2_attention(
  query_dim,
  heads = 8L,
  kv_heads = 8L,
  dim_head = 64L,
  dropout = 0,
  bias = TRUE,
  cross_attention_dim = NULL,
  out_bias = TRUE,
  qk_norm = "rms_norm_across_heads",
  norm_eps = 1e-06,
  norm_elementwise_affine = TRUE,
  rope_type = "interleaved"
)
}
\description{
LTX2 Attention module
}
\keyword{internal}
