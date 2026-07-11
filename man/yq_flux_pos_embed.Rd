% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_flux_pos_embed}
\alias{yq_flux_pos_embed}
\title{FLUX rotary position embedding tables (host-side)}
\usage{
yq_flux_pos_embed(ids, axes_dim = c(32L, 32L, 32L, 32L), theta = 2000,
                  device = "cpu")
}
\arguments{
\item{ids}{Numeric matrix \code{[S, n_axes]} of position ids.}

\item{axes_dim}{Integer vector of per-axis rotary dims (must sum to
the attention head dim; FLUX.2: \code{c(32, 32, 32, 32)}).}

\item{theta}{Numeric. Base frequency (FLUX.2: 2000).}

\item{device}{Character. Target device.}
}
\value{
List \code{list(cos, sin)}, each an AnvlArray
  \code{[S, sum(axes_dim)]}, f32.
}
\description{
Base-R port of \code{diffuseR::flux_pos_embed}: per-axis frequencies
(float64 outer products), cos/sin, each doubled by interleave, then
concatenated across axes. Deterministic, no weights.
}
