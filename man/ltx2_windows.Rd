% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx2_windows}
\alias{ltx2_windows}
\title{Fixed windows over a narration}
\usage{
ltx2_windows(frames, window = 97L, carry = 25L)
}
\arguments{
\item{frames}{Frames to cover.}

\item{window}{Frames per window, 8k+1 (default 97).}

\item{carry}{Overlap in frames, 8k+1, at least 9 and below
\code{window} (default 25, four latent frames).}
}
\value{
Data frame with \code{index}, \code{start} (0-based frame)
  and \code{frames}.
}
\description{
Lays out windows of \code{window} frames that overlap by \code{carry}
frames and cover \code{frames}. Every window but the last is
\code{window} frames; the last is sized up to the end on the 8k+1
grid and is never shorter than \code{carry + 8}.
}
\examples{
ltx2_windows(300)

}
