% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_encode_video_frames}
\alias{ltx23_encode_video_frames}
\title{Encode pixel frames to normalized video latents}
\usage{
ltx23_encode_video_frames(vae, frames)
}
\arguments{
\item{vae}{An \code{ltx23_video_vae}.}

\item{frames}{Tensor [1, 3, F, H, W] in [-1, 1] (see
\code{\link{ltx23_preprocess_frames}}).}
}
\value{
Normalized latents [1, 128, F', H/32, W/32] (float32).
}
\description{
VAE encode in "argmax" mode (the distribution mean), then normalize
with the checkpoint's per-channel statistics — the exact inverse of
the decode path.
}
