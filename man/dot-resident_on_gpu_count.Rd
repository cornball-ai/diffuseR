% tinyrox says don't edit this manually, but it can't stop you!
\name{.resident_on_gpu_count}
\alias{.resident_on_gpu_count}
\title{How many components actually have their tensors on the GPU}
\usage{
.resident_on_gpu_count(staging)
}
\arguments{
\item{staging}{A named list of staging sets.}
}
\value{
Integer. Number of components whose live tensors are on CUDA.
}
\description{
Ground truth, as opposed to the handle's declared state. The two can
disagree: a pipeline built with \code{phase_offload = TRUE} swaps each
component back to pinned host memory as its phase finishes, so after a
render the handle is still "active" while the card holds nothing. A
broker deciding who to evict needs the measurement, not the claim.
}
\keyword{internal}
