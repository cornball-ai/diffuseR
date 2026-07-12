% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_zimage_time_embed}
\alias{yq_zimage_time_embed}
\title{Z-Image timestep sinusoid (host-side)}
\usage{
yq_zimage_time_embed(t, freq_size = 256L, t_scale = 1000, max_period = 10000,
                     device = "cpu")
}
\arguments{
\item{t}{Numeric vector of timesteps in \code{[0, 1]}.}

\item{freq_size}{Integer. Sinusoid width (256).}

\item{t_scale}{Numeric. Timestep scale (1000).}

\item{max_period}{Numeric. Base period (10000).}

\item{device}{Character. Target device.}
}
\value{
AnvlArray \code{[length(t), freq_size]}, f32.
}
\description{
Parameter-free sinusoidal embedding of \code{t * t_scale}, cos-first
(\code{flip_sin_to_cos = TRUE}, \code{downscale_freq_shift = 0}),
matching \code{diffuseR::ltx23_get_timestep_embedding} as called by the
Z-Image \code{t_embedder}. Computed in base R and returned as an
\code{AnvlArray}, mirroring how the RoPE tables are precomputed outside
the model.
}
