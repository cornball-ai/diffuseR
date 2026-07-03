% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_per_token_rms_norm}
\alias{ltx23_per_token_rms_norm}
\title{Per-token RMS normalization over the channel axis}
\usage{
ltx23_per_token_rms_norm(x, eps = 1e-06)
}
\arguments{
\item{x}{Tensor [B, S, C, L] of stacked per-layer hidden states.}

\item{eps}{Numeric. Stability epsilon.}
}
\value{
Tensor of the same shape.
}
\description{
Per-token RMS normalization over the channel axis
}
