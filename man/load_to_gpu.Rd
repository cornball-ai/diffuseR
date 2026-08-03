% tinyrox says don't edit this manually, but it can't stop you!
\name{load_to_gpu}
\alias{load_to_gpu}
\title{Load Module to GPU}
\usage{
load_to_gpu(module, device = "cuda")
}
\arguments{
\item{module}{A torch nn_module.}

\item{device}{Character. Target device (default "cuda").}
}
\value{
The module (modified in place).
}
\description{
Moves a torch module and all its parameters to CUDA.
}
\examples{
if (torch::torch_is_installed()) {
  model <- torch::nn_linear(4, 2)
  # "cuda" needs a GPU; "cpu" is the portable round trip.
  load_to_gpu(model, device = "cpu")
}
}
