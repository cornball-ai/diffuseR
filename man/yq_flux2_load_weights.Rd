% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_flux2_load_weights}
\alias{yq_flux2_load_weights}
\title{Load FLUX.2 Klein transformer weights into an anvl pytree}
\usage{
yq_flux2_load_weights(path, num_layers = 5L, num_single_layers = 20L,
                      device = "cpu")
}
\arguments{
\item{path}{Path to \code{transformer/diffusion_pytorch_model.safetensors}.}

\item{num_layers}{Integer. Double-stream blocks (5).}

\item{num_single_layers}{Integer. Single-stream blocks (20).}

\item{device}{Character. Target device.}
}
\value{
Weights pytree.
}
\description{
Reads every transformer weight from the checkpoint (bf16 upcast to
f32), transposing 2-D linears to \code{[in, out]}, and wraps each as
an \code{AnvlArray} on \code{device} — freeing the R copy as it goes,
so peak host memory stays near one tensor rather than the full 15.5
GB twice. Returns the nested list \code{\link{yq_flux2_transformer}}
expects.
}
