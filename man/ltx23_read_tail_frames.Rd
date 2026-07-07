% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_read_tail_frames}
\alias{ltx23_read_tail_frames}
\title{Read the trailing frames of a video file}
\usage{
ltx23_read_tail_frames(path, n = 9L)
}
\arguments{
\item{path}{Video file.}

\item{n}{Integer. Trailing frame count.}
}
\value{
Array [n, H, W, 3] in [0, 1].
}
\description{
Extracts the last \code{n} frames of an MP4 (via \code{av}) for use
as continuation conditioning.
}
