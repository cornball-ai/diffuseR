% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_flux1_load_weights}
\alias{yq_flux1_load_weights}
\title{Load FLUX.1 transformer weights into an anvl pytree}
\usage{
yq_flux1_load_weights(path, num_layers = 19L, num_single_layers = 38L,
                      device = "cpu")
}
\arguments{
\item{path}{Path to the safetensors file holding the state dict.}

\item{num_layers}{Integer. Double-stream blocks.}

\item{num_single_layers}{Integer. Single-stream blocks.}

\item{device}{Character. Target device.}
}
\value{
Weights pytree.
}
\description{
Reads every FLUX.1 transformer weight from a safetensors file (f32),
transposing 2-D linears to \code{[in, out]} and keeping biases / norm
weights as vectors, and wraps each as an \code{AnvlArray} on
\code{device}. The file is the random-init state-dict fixture written
by \code{tools/gen_fixture_flux1_dit.R} (which also carries the
inputs/output under \code{input.*}/\code{output}, ignored here);
the same loader works on a real diffusers checkpoint (identical key
tree). Returns the nested list \code{\link{yq_flux1_transformer}}
expects. Strict census: every named key must exist.
}
