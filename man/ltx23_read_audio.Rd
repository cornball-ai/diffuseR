% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_read_audio}
\alias{ltx23_read_audio}
\title{Read an audio file as 16 kHz stereo PCM}
\usage{
ltx23_read_audio(path, sample_rate = 16000L)
}
\arguments{
\item{path}{Audio file.}

\item{sample_rate}{Integer.}
}
\value{
Matrix [2, samples] in [-1, 1].
}
\description{
Decodes MP3/WAV/etc. via \code{av} to 16-bit PCM at the target rate
and parses the RIFF container in base R.
}
