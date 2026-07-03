% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_upsample1d}
\alias{ltx23_upsample1d}
\title{Anti-aliasing 1D upsampler (transposed low-pass)}
\usage{
ltx23_upsample1d(ratio = 2L, kernel_size = NULL, window_type = "kaiser")
}
\arguments{
\item{ratio}{Integer. Upsampling ratio.}

\item{kernel_size}{Integer or NULL.}

\item{window_type}{"kaiser" (BigVGAN default) or "hann" (final resampler).}
}
\description{
Anti-aliasing 1D upsampler (transposed low-pass)
}
