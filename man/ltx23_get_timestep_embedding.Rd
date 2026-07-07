% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_get_timestep_embedding}
\alias{ltx23_get_timestep_embedding}
\title{Sinusoidal timestep embedding}
\usage{
ltx23_get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos = TRUE,
                             downscale_freq_shift = 0, max_period = 10000)
}
\arguments{
\item{timesteps}{1D tensor of timestep values.}

\item{embedding_dim}{Integer. Output embedding size.}

\item{flip_sin_to_cos}{Logical. Put cos before sin.}

\item{downscale_freq_shift}{Numeric. Frequency delta control.}

\item{max_period}{Numeric. Maximum embedding frequency period.}
}
\value{
Tensor [N, embedding_dim].
}
\description{
DDPM-style sinusoidal embedding. LTX uses \code{flip_sin_to_cos=TRUE}
(cos first) and \code{downscale_freq_shift=0}.
}
