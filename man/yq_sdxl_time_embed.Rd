% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_sdxl_time_embed}
\alias{yq_sdxl_time_embed}
\title{Host-side sinusoidal timestep embedding for the SDXL UNet}
\usage{
yq_sdxl_time_embed(timestep, dim = 320L, device = "cpu")
}
\arguments{
\item{timestep}{Numeric vector \code{[batch]} of timesteps.}

\item{dim}{Integer. Embedding width (SDXL main embed: 320, even).}

\item{device}{Character. Target device.}
}
\value{
AnvlArray \code{[batch, dim]}, f32.
}
\description{
Precomputes the parameter-free sinusoidal embedding (the input to the
UNet's time-embedding MLP), mirroring \code{diffuseR}'s
\code{timestep_embedding(timestep, dim, flip_sin_to_cos = TRUE,
downscale_freq_shift = 0)}: \code{[cos(args), sin(args)]} with
\code{args = t * exp(-log(10000) / half_dim * (0:(half_dim-1)))}.
Computed in base R and returned as an \code{AnvlArray} so the jit
boundary stays weight-only.
}
