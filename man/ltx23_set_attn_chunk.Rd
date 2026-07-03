% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_set_attn_chunk}
\alias{ltx23_set_attn_chunk}
\title{Set the attention query-chunk size across a transformer}
\usage{
ltx23_set_attn_chunk(transformer, chunk)
}
\arguments{
\item{transformer}{An \code{ltx23_transformer}.}

\item{chunk}{Integer or NULL.}
}
\value{
Invisibly, the transformer.
}
\description{
R torch has no fused attention, so the [B, H, S, S] matrix
materializes; chunking queries bounds the peak. NULL disables
chunking.
}
