% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_tail_latents}
\alias{ltx23_tail_latents}
\title{Slice the trailing latent frames of a generation for chaining}
\usage{
ltx23_tail_latents(result, k = 2L, latent_shape = NULL)
}
\arguments{
\item{result}{A \code{\link{txt2vid_ltx2}} result list (uses its
\code{latents} and \code{latent_shape}), or the packed latents
tensor [1, S, 128] itself (then \code{latent_shape} is
required).}

\item{k}{Integer. Trailing latent frames to keep (default 2 = the
standard 9-pixel-frame conditioning prefix).}

\item{latent_shape}{Integer vector c(frames, height, width) of the
latent geometry; only needed when \code{result} is a raw tensor.}
}
\value{
Normalized latents [1, 128, k, H', W'] (float32), ready
  for \code{txt2vid_ltx2(condition_latents = )}.
}
\description{
Cuts the last \code{k} latent frames out of a result's video
latents, in the [1, 128, k, H', W'] layout that
\code{txt2vid_ltx2(condition_latents = )} consumes, so one chunk
can seed the next without leaving latent space: no decode, no
re-encode, no video round-trip.
}
\details{
Semantics caveat: a latent frame sliced from inside a sequence
represents 8 pixel frames, while a fresh VAE encode of a k-frame
tail represents 1 + 8(k - 1) pixel frames with its first latent in
first-frame form. The frozen prefix the next generation sees is
therefore not identical to the pixel-path prefix; compare both on
real content before relying on latent-only joins.

}
