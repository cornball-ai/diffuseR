% tinyrox says don't edit this manually, but it can't stop you!
\name{write_wav}
\alias{write_wav}
\title{Write a 16-bit PCM WAV file}
\usage{
write_wav(audio, path, sample_rate = 48000L)
}
\arguments{
\item{audio}{Numeric matrix [channels, samples] in [-1, 1].}

\item{path}{Output path.}

\item{sample_rate}{Integer.}
}
\value{
Invisibly, the path.
}
\description{
Minimal RIFF writer in base R.
}
