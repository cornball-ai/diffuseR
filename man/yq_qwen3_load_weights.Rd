% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_qwen3_load_weights}
\alias{yq_qwen3_load_weights}
\title{Load Qwen3-4B text-encoder weights for FLUX.2 into an anvl pytree}
\usage{
yq_qwen3_load_weights(dir, n_layers = 27L, device = "cpu")
}
\arguments{
\item{dir}{The \code{text_encoder} directory (index + shards).}

\item{n_layers}{Integer. Decoder layers to load (klein: 27, enough
for out_layers up to 27).}

\item{device}{Character. Target device.}
}
\value{
List \code{list(embed = <R matrix [vocab, hidden]>,
  layers = <list of per-layer weight lists>)}.
}
\description{
Reads the sharded \code{text_encoder} checkpoint (bf16 upcast to f32).
The embedding table stays an R matrix for host-side gather
(\code{\link{yq_qwen3_embed}}); only the first \code{n_layers} decoder
layers are loaded (klein consumes mid-stack states, so the deeper
layers and the tied LM head are never needed). Each tensor is wrapped
as an \code{AnvlArray} as it is read, freeing the R copy.
}
