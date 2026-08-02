% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_mel_stft}
\alias{ltx23_mel_stft}
\title{Causal log-mel spectrogram with checkpoint-loaded bases}
\usage{
ltx23_mel_stft(filter_length = 512L, hop_length = 80L, window_length = 512L,
               num_mel_channels = 64L)
}
\arguments{
\item{filter_length,hop_length,window_length,num_mel_channels}{Integers.}
}
\value{
Module whose forward(waveform) returns the log-mel
  spectrogram [B, n_mels, frames], clamped at 1e-5 before the log.
}
\description{
Causal log-mel spectrogram with checkpoint-loaded bases
}
