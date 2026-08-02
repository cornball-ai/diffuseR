% tinyrox says don't edit this manually, but it can't stop you!
\name{.resident_pin}
\alias{.resident_pin}
\title{Pin every component of a pipeline for fast transfer}
\usage{
.resident_pin(pipeline, verbose = TRUE)
}
\arguments{
\item{pipeline}{A loaded diffuseR pipeline.}

\item{verbose}{Print progress.}
}
\value{
A named list of staging sets, one per component that could be
  pinned. Components that fail to page-lock are absent, and fall back
  to the pageable \code{$to()} path.
}
\description{
Re-uses any staging the loader already built (the phase-offload path
pins as part of loading), and pins the rest. Pinning a component that
is currently on the GPU also evicts it, since \code{.pin_component}
copies into page-locked host memory and re-points the live tensors at
it, so this doubles as the initial offload.
}
\keyword{internal}
