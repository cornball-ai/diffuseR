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
\dontrun{
load_to_gpu(model)
output <- model(x)
offload_to_cpu(model)
}
}
