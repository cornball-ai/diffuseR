% tinyrox says don't edit this manually, but it can't stop you!
\name{.pin_component}
\alias{.pin_component}
\title{Pin a component's tensors for fast phase transfer}
\usage{
.pin_component(module, extra = NULL)
}
\arguments{
\item{module}{An nn_module on the CPU.}

\item{extra}{Optional list of additional plain-field tensors to pin
alongside the module's parameters and buffers (e.g. an fp8
linear's \code{weight_fp8}/\code{weight_scale} fields, which live
outside \code{parameters}/\code{buffers}). \code{set_data} mutates
each tensor in place, so the field reference stays valid.}
}
\value{
A list of \code{list(live, pinned)} tensor pairs, or NULL
  if pinning is unavailable (no CUDA, or page-locking failed).
}
\description{
Pin a component's tensors for fast phase transfer
}
\keyword{internal}
