% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_audio_mel_frontend}
\alias{ltx23_audio_mel_frontend}
\title{Build the 16 kHz log-mel frontend for audio conditioning}
\usage{
ltx23_audio_mel_frontend(filter_length = 1024L, hop_length = 160L,
                         n_mels = 64L, sample_rate = 16000L, fmin = 0,
                         fmax = 8000)
}
\arguments{
\item{filter_length,hop_length,n_mels,sample_rate,fmin,fmax}{The
checkpoint preprocessing parameters (defaults are LTX-2.3's).}
}
\value{
An \code{ltx23_mel_stft} module.
}
\description{
An \code{\link{ltx23_mel_stft}} whose STFT and mel bases are
constructed (not checkpoint-loaded) with the audio VAE's
preprocessing spec.
}
