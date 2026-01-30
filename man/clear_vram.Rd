% tinyrox says don't edit this manually, but it can't stop you!
\name{clear_vram}
\alias{clear_vram}
\title{Clear VRAM Cache}
\usage{
clear_vram(verbose = FALSE)
}
\arguments{
\item{verbose}{Logical. Print memory status before/after.}
}
\value{
Invisibly returns NULL.
}
\description{
Forces garbage collection and clears CUDA memory cache.
}
\examples{
\dontrun{
clear_vram()
}
}
