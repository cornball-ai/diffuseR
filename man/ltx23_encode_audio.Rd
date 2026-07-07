% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_encode_audio}
\alias{ltx23_encode_audio}
\title{Encode audio into normalized, packed conditioning latents}
\usage{
ltx23_encode_audio(audio_vae, wav, audio_num_frames, frontend = NULL)
}
\arguments{
\item{audio_vae}{An \code{ltx23_audio_vae} (with encoder weights).}

\item{wav}{Matrix [2, samples] in [-1, 1] at 16 kHz (see
\code{\link{ltx23_read_audio}}).}

\item{audio_num_frames}{Integer. Target latent length.}

\item{frontend}{Optional prebuilt \code{\link{ltx23_audio_mel_frontend}}.}
}
\value{
Packed normalized latents [1, audio_num_frames, 128] (float32).
}
\description{
Pads or trims the waveform so the latent length equals
\code{audio_num_frames} (mel frames \code{4L - 3}, mirroring the
decoder's \code{target_frames}), computes the log-mel, encodes in
argmax mode, packs, and normalizes with the checkpoint statistics.
}
