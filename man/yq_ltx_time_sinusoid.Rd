% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_ltx_time_sinusoid}
\alias{yq_ltx_time_sinusoid}
\title{LTX-2.3 timestep sinusoid (host-side)}
\usage{
yq_ltx_time_sinusoid(t, dim = 256L, max_period = 10000, device = "cpu")
}
\arguments{
\item{t}{Numeric vector of timestep values (row-major flatten of the
per-token \code{[B, S]} timestep).}

\item{dim}{Integer. Sinusoid width (256).}

\item{max_period}{Numeric. Base period (10000).}

\item{device}{Character. Target device.}
}
\value{
AnvlArray \code{[length(t), dim]}, f32.
}
\description{
Parameter-free sinusoidal embedding matching
\code{diffuseR::ltx23_get_timestep_embedding} (cos-first,
\code{downscale_freq_shift = 0}, \code{max_period = 10000}). Computed
in base R (deterministic, no weights) and returned as an
\code{AnvlArray}, mirroring how the RoPE tables are precomputed outside
the model. Timestep values are NOT scaled here (the LTX pipeline scales
before calling the model).
}
