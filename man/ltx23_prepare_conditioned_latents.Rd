% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_prepare_conditioned_latents}
\alias{ltx23_prepare_conditioned_latents}
\title{Build conditioned initial latents and the conditioning mask}
\usage{
ltx23_prepare_conditioned_latents(cond_latents, latent_frames, latent_height,
                                  latent_width, noise, cond_noise_scale = 0)
}
\arguments{
\item{cond_latents}{Normalized condition latents
[1, 128, k, H', W'] from \code{\link{ltx23_encode_video_frames}}.}

\item{noise}{Tensor [1, 128, F', H', W'] of standard noise (caller
provides so seeding stays in one place).}

\item{cond_noise_scale}{Numeric. Optional partial noising of the
conditioned tokens (diffusers \code{noise_scale}, default 0).}

\item{latent_frames,latent_height,latent_width}{Integers. Full
latent geometry of the generation.}
}
\value{
list(latents [1, S, 128] float32 packed,
  conditioning_mask [1, S] float32 packed).
}
\description{
i2v (\code{cond_latents} has one latent frame): the encoded frame is
repeated across all latent frames and only latent frame 0 is marked
conditioned. Continuation (k latent frames): the prefix tokens are
replaced and marked. Unconditioned positions start as pure noise.
}
