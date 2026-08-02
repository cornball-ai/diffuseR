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
  without CUDA), \code{loaded_at}, and \code{last_error} (NULL unless
  a transition failed).
}
\description{
Status of a resident handle
}
