% tinyrox says don't edit this manually, but it can't stop you!
\name{save_video}
\alias{save_video}
\title{Save Video to File}
\usage{
save_video(
  video,
  file,
  fps = 24,
  format = NULL,
  backend = "auto",
  quality = 85,
  verbose = TRUE
)
}
\arguments{
\item{video}{Array of video frames with shape [T, H, W, C] where C is 3 (RGB).
Values should be in [0, 1] range.}

\item{file}{Character. Output file path. Extension determines format.}

\item{fps}{Numeric. Frames per second (default 24).}

\item{format}{Character. Output format: "mp4", "gif", "webm", or "frames".
If NULL, inferred from file extension.}

\item{backend}{Character. Backend to use: "ffmpeg", "av", or "auto".}

\item{quality}{Integer. Quality level 1-100 (for lossy formats).}

\item{verbose}{Logical. Print progress messages.}
}
\value{
Invisibly returns the output file path.
}
\description{
Saves a video array to a file in various formats.
}
\examples{
\dontrun{
# Save as MP4
save_video(video_array, "output.mp4", fps = 24)

# Save as GIF
save_video(video_array, "output.gif", fps = 10)

# Save as individual frames
save_video(video_array, "frames/", format = "frames")
}
}
