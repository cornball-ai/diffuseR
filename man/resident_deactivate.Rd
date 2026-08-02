% tinyrox says don't edit this manually, but it can't stop you!
\name{resident_deactivate}
\alias{resident_deactivate}
\title{Release a resident pipeline's VRAM}
\usage{
resident_deactivate(res, release = TRUE)
}
\arguments{
\item{res}{A \code{diffuseR_resident} handle.}

\item{release}{Empty the CUDA caching allocator afterwards. Leave TRUE
unless another handle on the same device is about to reuse the pool.}
}
\value{
Invisibly the handle, with state "inactive".
}
\description{
Re-points every component at its pinned host copy and drops the GPU
storage. Weights are immutable during inference, so the pinned copies
are still current and this moves no bytes: it is a pointer swap plus a
cache release. The handle stays loaded and can be reactivated without
touching the disk.
}
