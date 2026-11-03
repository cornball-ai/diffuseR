% tinyrox says don't edit this manually, but it can't stop you!
\name{.ltx23_keyframe_state}
\alias{.ltx23_keyframe_state}
\title{Build the conditioned starting state from keyframes}
\usage{
.ltx23_keyframe_state(
  keyframes,
  latent_frames,
  latent_height,
  latent_width,
  noise,
  encode,
  height,
  width,
  rope,
  fps,
  cond_noise_scale = 0,
  base = NULL,
  noise_scale = 1
)
}
\arguments{
\item{keyframes}{List of \code{\link{ltx23_keyframe}} objects.}

\item{noise}{Standard noise [1, 128, F', H', W'].}

\item{encode}{Function taking preprocessed frames to latents (only
called for pixel keyframes).}

\item{rope}{The transformer's video RoPE (for appended coordinates).}

\item{fps}{Frame rate.}

\item{cond_noise_scale}{Partial re-noising of fully pinned frame-0
tokens (the \code{txt2vid_ltx2} argument of that name).}

\item{base}{Optional normalized latents [1, 128, F', H', W'] to
re-enter from (two-stage: the upsampled stage-1 result). NULL starts
from pure noise.}

\item{noise_scale}{Entry noise level when \code{base} is given: each
token is re-noised by \code{(1 - strength) * noise_scale}.}

\item{latent_frames,latent_height,latent_width}{Latent geometry.}

\item{height,width}{Pixel size of the generation.}
}
\value{
list(latents [1, S + A, 128], conditioning_mask [1, S + A]
  strengths (0 = free), clean [1, S + A, 128], extra_coords [1, 3, A,
  2] or NULL, n_main = S).
}
\description{
Build the conditioned starting state from keyframes
}
\keyword{internal}
