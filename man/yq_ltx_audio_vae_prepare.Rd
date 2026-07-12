% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_ltx_audio_vae_prepare}
\alias{yq_ltx_audio_vae_prepare}
\title{Prepare packed audio latents for the VAE decoder (de-normalize + unpack)}
\usage{
yq_ltx_audio_vae_prepare(z_packed, latents_mean, latents_std, num_mel_bins)
}
\arguments{
\item{z_packed}{AnvlArray \code{[B, T, C*M]} packed audio latents.}

\item{num_mel_bins}{Integer. Latent mel bins \code{M} (\code{C*M} splits
into \code{(C, M)} row-major).}

\item{latents_mean,latents_std}{Numeric vectors \code{[C*M]}.}
}
\value{
AnvlArray \code{[B, C, T, M]}, de-normalized and unpacked.
}
\description{
Mirrors the pipeline's pre-decode step (\code{.ltx23_denormalize_audio}
then \code{ltx23_unpack_audio_latents}): applies the per-channel affine
\code{z * std + mean} on the packed \code{[B, T, C*M]} representation
(stats broadcast over the trailing feature dim), then reshapes to the
decoder's \code{[B, C, T, M]} input. The decoder's own \code{decode()}
applies no further normalization.
}
