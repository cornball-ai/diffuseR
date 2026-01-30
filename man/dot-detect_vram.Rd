% tinyrox says don't edit this manually, but it can't stop you!
\name{.detect_vram}
\alias{.detect_vram}
\title{Detect Available VRAM}
\usage{
.detect_vram(use_free = FALSE)
}
\arguments{
\item{use_free}{Logical. If TRUE, return free VRAM. If FALSE, return total.}
}
\value{
Numeric. VRAM in GB, or 0 if no GPU detected.
}
\description{
Uses gpuctl if available.
}
\keyword{internal}
