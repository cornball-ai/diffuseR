% tinyrox says don't edit this manually, but it can't stop you!
\name{save_video}
\alias{save_video}
\title{Save Video to File}
\usage{
save_video(video, file, fps = 24, format = NULL, backend = "auto",
           quality = 85, verbose = TRUE)
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
video <- array(runif(4 * 16 * 16 * 3), dim = c(4, 16, 16, 3))

# Individual PNG frames need no external encoder.
frame_dir <- file.path(tempdir(), "frames")
save_video(video, frame_dir, format = "frames", verbose = FALSE)
length(list.files(frame_dir, pattern = "[.]png$"))
unlink(frame_dir, recursive = TRUE)

# MP4 and GIF need an ffmpeg binary or the 'av' package.
\donttest{
if (requireNamespace("av", quietly = TRUE)) {
  out <- file.path(tempdir(), "output.mp4")
  save_video(video, out, fps = 24, verbose = FALSE)
  unlink(out)
}
}
}
