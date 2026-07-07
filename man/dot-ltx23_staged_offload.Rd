% tinyrox says don't edit this manually, but it can't stop you!
\name{.ltx23_staged_offload}
\alias{.ltx23_staged_offload}
\title{Return a pinned component to the CPU}
\usage{
.ltx23_staged_offload(staging)
}
\description{
Weights are immutable during inference, so the pinned host copies
are still current: offload is a pointer swap, no transfer.
}
\keyword{internal}
