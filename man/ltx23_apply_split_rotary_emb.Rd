% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_apply_split_rotary_emb}
\alias{ltx23_apply_split_rotary_emb}
\title{Apply split rotary embeddings}
\usage{
ltx23_apply_split_rotary_emb(x, freqs)
}
\arguments{
\item{x}{Tensor of shape [B, H, T, D] (per-head layout), or [B, T, H*D]
which is reshaped per-head when \code{freqs} is 4D.}

\item{freqs}{List of two tensors (cos, sin), each [B, H, T, D/2].}
}
\value{
Tensor with the same shape and dtype as \code{x}.
}
\description{
Rotates element pairs formed by splitting the last dimension in half:
element i pairs with element i + d/2. The cos/sin tensors carry half
the head dimension.
}
