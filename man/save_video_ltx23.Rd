% tinyrox says don't edit this manually, but it can't stop you!
\name{save_video_ltx23}
\alias{save_video_ltx23}
\title{Save an LTX video (optionally with audio) to MP4}
\usage{
save_video_ltx23(video, filename, fps = 24, audio = NULL, sample_rate = 48000L,
                 verbose = TRUE)
}
\arguments{
\item{video}{Array [frames, height, width, 3] in [0, 1].}

\item{filename}{Output path (.mp4).}

\item{fps}{Numeric.}

\item{audio}{Optional numeric matrix [channels, samples] in [-1, 1].}

\item{sample_rate}{Integer.}

\item{verbose}{Logical.}
}
\value{
Invisibly, the filename.
}
\description{
Uses the av package (Suggests) to encode frames and mux the audio
track.
}
