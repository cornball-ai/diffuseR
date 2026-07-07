% tinyrox says don't edit this manually, but it can't stop you!
\name{flux2_empirical_mu}
\alias{flux2_empirical_mu}
\title{Empirical timestep shift for FLUX.2}
\usage{
flux2_empirical_mu(image_seq_len, num_steps)
}
\arguments{
\item{image_seq_len}{Integer. Packed image token count.}

\item{num_steps}{Integer. Inference steps.}
}
\value{
Numeric mu for \code{\link{flowmatch_set_timesteps}}.
}
\description{
BFL's piecewise-linear fit of the dynamic-shifting mu as a function of
image sequence length and step count; replaces FLUX.1's
calculate_shift. Reference: compute_empirical_mu (adapted from BFL
sampling.py).
}
