% tinyrox says don't edit this manually, but it can't stop you!
\name{flux_ada_layer_norm_zero}
\alias{flux_ada_layer_norm_zero}
\title{FLUX adaLN-Zero modulation (double-stream)}
\usage{
flux_ada_layer_norm_zero(dim)
}
\arguments{
\item{dim}{Integer. Model dimension.}
}
\value{
Module whose forward(x, emb) returns
  \code{list(x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp)}.
}
\description{
Projects the conditioning embedding to six modulation vectors and
returns the msa-modulated input plus the remaining parameters.
Reference: diffusers AdaLayerNormZero.
}
