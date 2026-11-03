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
  cond_noise_scale = 0
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
