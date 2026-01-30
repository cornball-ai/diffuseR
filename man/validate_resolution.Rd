% tinyrox says don't edit this manually, but it can't stop you!
\name{validate_resolution}
\alias{validate_resolution}
\title{Validate Resolution Against Profile}
\usage{
validate_resolution(height, width, num_frames, profile)
}
\arguments{
\item{height}{Integer. Requested height.}

\item{width}{Integer. Requested width.}

\item{num_frames}{Integer. Requested number of frames.}

\item{profile}{Memory profile from `ltx2_memory_profile()`.}
}
\value{
List with adjusted height, width, num_frames and warning if adjusted.
}
\description{
Checks if requested resolution fits within memory profile limits.
}
\examples{
\dontrun{
profile <- ltx2_memory_profile(vram_gb = 8)
validated <- validate_resolution(720, 1280, 60, profile)
}
}
