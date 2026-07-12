% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_gemma3_load_weights}
\alias{yq_gemma3_load_weights}
\title{Load Gemma3 text-encoder weights into an anvl pytree}
\usage{
yq_gemma3_load_weights(path, num_layers = 6L, device = "cpu")
}
\arguments{
\item{path}{Path to a \code{.safetensors} file holding the state_dict
(keys \code{embed_tokens.weight}, \code{layers.\{i\}.*}, \code{norm.weight};
layer indices 0-based, as HF / R torch emit them).}

\item{num_layers}{Integer. Decoder layers to load.}

\item{device}{Character. Target device.}
}
\value{
List \code{list(embed = <R matrix [vocab, hidden]>,
  layers = <per-layer weight lists>, norm = <AnvlArray>)}.
}
\description{
Reads the state_dict from a single safetensors file (the parity
fixture, or a real checkpoint), transposing 2-D linears to
\code{[in, out]} and adding 1 to every RMSNorm weight host-side so
plain \code{\link[yunque]{yq_rms_norm}} reproduces Gemma's
\code{(1 + weight)} form. The embedding table stays an R matrix for
host-side gather (\code{\link{yq_gemma3_embed}}); every other tensor is
wrapped as an \code{AnvlArray} on \code{device}.
}
