% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_kaiser_sinc_filter1d}
\alias{ltx23_kaiser_sinc_filter1d}
\title{Kaiser sinc low-pass filter kernel}
\usage{
ltx23_kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
}
\arguments{
\item{cutoff}{Numeric. Normalized cutoff in (0, 0.5].}

\item{half_width}{Numeric. Transition band half width.}

\item{kernel_size}{Integer.}
}
\value{
Tensor [kernel_size].
}
\description{
Kaiser sinc low-pass filter kernel
}
