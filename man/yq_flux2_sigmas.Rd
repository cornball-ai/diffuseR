% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_flux2_sigmas}
\alias{yq_flux2_sigmas}
\title{FLUX.2 Klein FlowMatch sigma schedule (host-side)}
\usage{
yq_flux2_sigmas(seq_img, n_steps = 4L)
}
\arguments{
\item{seq_img}{Integer. Packed image token count.}

\item{n_steps}{Integer. Denoising steps (klein: 4).}
}
\value{
Numeric vector of length \code{n_steps + 1}: the shifted
  sigmas followed by 0. \code{sigmas[i]} is both the DiT timestep at
  step i and the ODE position; \code{dt = sigmas[i+1] - sigmas[i]}.
}
\description{
Dynamic exponential timestep shift with the BFL empirical mu, plus the
terminal-0 sigma. Matches diffusers' \code{flowmatch_set_timesteps}
with \code{use_dynamic_shifting = TRUE}, \code{sigmas = seq(1, 1/n,
length.out = n)}. Distilled klein is guidance-free.
}
