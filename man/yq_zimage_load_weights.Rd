% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_zimage_load_weights}
\alias{yq_zimage_load_weights}
\title{Load Z-Image DiT weights into an anvl pytree}
\usage{
yq_zimage_load_weights(path, patch_key = "2-1", device = "cpu")
}
\arguments{
\item{path}{Path to the state-dict safetensors file.}

\item{patch_key}{Character. Embedder / final-layer module-dict key.}

\item{device}{Character. Target device.}
}
\value{
Weights pytree.
}
\description{
Reads a \code{zimage_transformer} state-dict safetensors file (f32),
transposing 2-D linears to \code{[in, out]} and wrapping each tensor as
an \code{AnvlArray} on \code{device}. Block counts are derived from the
key census (\code{layers.N}, \code{noise_refiner.N},
\code{context_refiner.N}). Returns the nested list
\code{\link{yq_zimage_dit}} expects. The patch key is \code{"2-1"}
(patch_size 2, f_patch_size 1).
}
