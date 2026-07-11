% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_flux2_time_proj}
\alias{yq_flux2_time_proj}
\title{FLUX.2 Klein timestep sinusoidal projection (host-side)}
\usage{
yq_flux2_time_proj(timestep, dim = 256L, max_period = 10000, device = "cpu")
}
\arguments{
\item{timestep}{Numeric vector (sigma space, 0-1); scaled by 1000
internally.}

\item{dim}{Integer. Projection width (klein: 256).}

\item{max_period}{Numeric. Base period (10000).}

\item{device}{Character. Target device.}
}
\value{
AnvlArray \code{[length(timestep), dim]}, f32.
}
\description{
Parameter-free sinusoidal embedding of \code{timestep * 1000},
matching diffusers \code{get_timestep_embedding} with
\code{flip_sin_to_cos = TRUE}, \code{downscale_freq_shift = 0}.
Computed in base R (deterministic, no weights) and returned as an
\code{AnvlArray}, mirroring how the RoPE tables are precomputed
outside the model.
}
