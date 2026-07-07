% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_apply_interleaved_rotary_emb}
\alias{ltx23_apply_interleaved_rotary_emb}
\title{Apply interleaved rotary embeddings}
\usage{
ltx23_apply_interleaved_rotary_emb(x, freqs)
}
\arguments{
\item{x}{Tensor of shape [B, S, C].}

\item{freqs}{List of two tensors (cos, sin), each [B, S, C].}
}
\value{
Tensor with the same shape and dtype as \code{x}.
}
\description{
Rotates adjacent element pairs of the last dimension:
\code{out = x * cos + rotate_half(x) * sin} with pairs interleaved
(elements 1,2 form the first complex pair).
}
