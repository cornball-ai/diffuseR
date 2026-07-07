% tinyrox says don't edit this manually, but it can't stop you!
\name{flux_apply_rotary_emb}
\alias{flux_apply_rotary_emb}
\title{Apply FLUX rotary embeddings to a per-head tensor}
\usage{
flux_apply_rotary_emb(x, freqs)
}
\arguments{
\item{x}{Tensor of shape [B, H, S, D] (per-head layout).}

\item{freqs}{List of two tensors (cos, sin), each [S, D], from
\code{flux_pos_embed}.}
}
\value{
Tensor with the same shape and dtype as \code{x}.
}
\description{
Rotates adjacent element pairs of the last dimension:
\code{out = x * cos + rotate_half(x) * sin} with pairs interleaved
(elements 1,2 form the first complex pair). Math in float32, result
cast back to the input dtype. Reference: apply_rotary_emb with
use_real_unbind_dim = -1.
}
