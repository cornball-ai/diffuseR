% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_keyframe}
\alias{ltx23_keyframe}
\title{Define an LTX keyframe}
\usage{
ltx23_keyframe(x, frame = 0L, strength = 1)
}
\arguments{
\item{x}{The content: normalized video latents [1, 128, k, H/32,
W/32] (e.g. from \code{\link{ltx23_tail_latents}} or
\code{\link{ltx23_encode_video_frames}}); an image path; an [H, W,
3] image array; or an [F, H, W, 3] array of frames in [0, 1]
(VAE-encoded at generation time, so F should be 8k + 1).}

\item{frame}{Integer. Pixel frame (0-based) the keyframe starts at.}

\item{strength}{Numeric in (0, 1]. 1 holds the content as given;
lower values let the generation move away from it.}
}
\value{
An \code{ltx23_keyframe} object.
}
\description{
A keyframe conditions an \code{\link{txt2vid_ltx2}} generation on
content at a frame: a first frame, an end frame, a carried tail from
a previous chunk, or any latent frames in between. Pass a list of
them as \code{txt2vid_ltx2(keyframes = )}.
}
\details{
At frame 0 the keyframe replaces the generation's first
latent frames and, at strength 1, decodes exactly as given there.
Anywhere else it is appended as extra tokens positioned at
\code{frame}: the generation attends to them and moves toward them,
so they act as a strong guide rather than a copy (at 512x512, a still
at frame 48 came back at about 25 dB PSNR). The last latent frame of
an 8k + 1 clip starts at pixel frame \code{num_frames - 8}.
}
