% tinyrox says don't edit this manually, but it can't stop you!
\name{flux_ada_layer_norm_zero_single}
\alias{flux_ada_layer_norm_zero_single}
\title{FLUX adaLN-Zero modulation (single-stream)}
\usage{
flux_ada_layer_norm_zero_single(dim)
}
\arguments{
\item{dim}{Integer. Model dimension.}
}
\value{
Module whose forward(x, emb) returns \code{list(x_norm, gate)}.
}
\description{
Three modulation vectors: shift, scale, gate. Reference: diffusers
AdaLayerNormZeroSingle.
}
