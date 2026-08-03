% tinyrox says don't edit this manually, but it can't stop you!
\name{resident_status}
\alias{resident_status}
\title{Status of a resident handle}
\usage{
resident_status(res)
}
\arguments{
\item{res}{A \code{diffuseR_resident} handle.}
}
\value{
A list with \code{model}, \code{state}, \code{device},
  \code{components} (character vector of pinned component names),
  \code{pinned_bytes} (page-locked host bytes held),
  \code{gpu_allocated} and \code{gpu_reserved} (bytes the CUDA
  caching allocator reports live and held for this process, NA
  without CUDA), \code{components_on_gpu} (how many components are
  *actually* resident right now), \code{loaded_at}, and
  \code{last_error} (NULL unless a transition failed).

  \code{state} is the handle's claim on the card;
  \code{components_on_gpu} is the measurement. They disagree by
  design after a render on a \code{phase_offload = TRUE} pipeline,
  which returns each component to pinned host memory as its phase
  finishes: the handle stays "active" (it still owns the card's
  budget and can render again without touching disk) while
  \code{components_on_gpu} is 0. Schedule on the measurement.
}
\description{
Status of a resident handle
}
