% tinyrox says don't edit this manually, but it can't stop you!
\name{save_video_ffmpeg}
\alias{save_video_ffmpeg}
\title{Save Video using FFmpeg}
\usage{
save_video_ffmpeg(
  video,
  file,
  fps = 24,
  format = "mp4",
  quality = 85,
  verbose = TRUE
)
}
\arguments{
\item{video}{Array of video frames [T, H, W, C].}

\item{file}{Output file path.}

\item{fps}{Frames per second.}

\item{format}{Output format.}

\item{quality}{Quality level 1-100.}

\item{verbose}{Logical.}
}
\description{
Save Video using FFmpeg
}
\keyword{internal}
