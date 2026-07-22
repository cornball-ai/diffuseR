% tinyrox says don't edit this manually, but it can't stop you!
\name{.ltx23_pin_component}
\alias{.ltx23_pin_component}
\title{Pin a component's tensors for fast phase transfer}
\usage{
.ltx23_pin_component(module)
}
\arguments{
\item{module}{An nn_module on the CPU.}
}
\value{
A list of \code{list(live, pinned)} tensor pairs, or NULL
  if pinning is unavailable (no CUDA, or page-locking failed).
}
\description{
Pin a component's tensors for fast phase transfer
}
\keyword{internal}
