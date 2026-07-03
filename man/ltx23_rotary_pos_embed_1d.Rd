% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_rotary_pos_embed_1d}
\alias{ltx23_rotary_pos_embed_1d}
\title{1D rotary embeddings for the text connectors}
\usage{
ltx23_rotary_pos_embed_1d(dim, base_seq_len = 4096L, theta = 10000,
                          double_precision = TRUE, rope_type = "split",
                          num_attention_heads = 32L)
}
\arguments{
\item{dim}{Integer. Rotary dimension (connector inner dim).}

\item{base_seq_len}{Integer. Base sequence length for normalization.}

\item{theta}{Numeric. RoPE theta.}

\item{double_precision}{Logical. Compute base frequencies in float64.}

\item{rope_type}{"split" (LTX-2.3) or "interleaved".}

\item{num_attention_heads}{Integer. For the split per-head layout.}
}
\description{
1D rotary embeddings for the text connectors
}
