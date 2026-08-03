% tinyrox says don't edit this manually, but it can't stop you!
\name{flux_ada_layer_norm_continuous}
\alias{flux_ada_layer_norm_continuous}
\title{FLUX continuous adaLN (final norm)}
\usage{
flux_ada_layer_norm_continuous(dim, cond_dim = dim, bias = TRUE)
}
\arguments{
\item{dim}{Integer. Model dimension.}

\item{cond_dim}{Integer. Conditioning embedding dimension.}

\item{bias}{Logical. Bias on the projection (TRUE for FLUX.1, FALSE
for FLUX.2).}
}
\value{
Module whose forward(x, cond) returns \code{x} normalized and
  then scaled and shifted by the conditioning embedding, a tensor of
  the same shape as \code{x}.
}
\description{
Scale/shift conditioning of the final norm. Note the chunk order:
scale first, then shift (the reverse of adaLN-Zero). Reference:
diffusers AdaLayerNormContinuous as used by FLUX norm_out
(elementwise_affine = FALSE, eps = 1e-6).
}
