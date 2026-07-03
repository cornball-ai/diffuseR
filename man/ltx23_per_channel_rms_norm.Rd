% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_per_channel_rms_norm}
\alias{ltx23_per_channel_rms_norm}
\title{Per-channel RMS normalization}
\usage{
ltx23_per_channel_rms_norm(eps = 1e-08)
}
\arguments{
\item{eps}{Numeric. Stability epsilon.}
}
\description{
Normalizes by the root-mean-square across the channel dimension
(dim 2 of [B, C, F, H, W]); no learned parameters.
}
