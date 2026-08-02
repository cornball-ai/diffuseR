% tinyrox says don't edit this manually, but it can't stop you!
\name{offload_to_cpu}
\alias{offload_to_cpu}
\title{Offload Module to CPU}
\usage{
offload_to_cpu(module, gc = TRUE)
}
\arguments{
\item{module}{A torch nn_module.}

\item{gc}{Logical. Run garbage collection after offload.}
}
\value{
The module (modified in place).
}
\description{
Moves a torch module and all its parameters to CPU.
}
\examples{
if (torch::torch_is_installed()) {
  model <- torch::nn_linear(4, 2)
  offload_to_cpu(model)
}
}
