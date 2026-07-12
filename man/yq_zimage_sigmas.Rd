% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_zimage_sigmas}
\alias{yq_zimage_sigmas}
\title{Z-Image-Turbo FlowMatch sigma schedule (host-side)}
\usage{
yq_zimage_sigmas(n_steps, shift = 3)
}
\arguments{
\item{n_steps}{Integer. Denoising steps (Turbo: 8).}

\item{shift}{Numeric. Static schedule shift (3.0).}
}
\value{
List: \code{sigmas} (length n, shifted), \code{sigmas_full}
  (sigmas with the terminal 0 appended), \code{t_model} (1 - sigmas,
  the reversed normalized timestep fed to the DiT).
}
\description{
The checkpoint scheduler's static exponential shift on the linear
sigma ramp \code{seq(1, 1/n, n)} (no dynamic shifting; Turbo's
calculate_shift/mu path is dead for this model). Mirrors
\code{flowmatch_set_timesteps(shift = 3, sigmas = seq(1, 1/n, n))}.
The model consumes the reversed normalized timestep
\code{t_model = 1 - sigma} and its velocity is negated in the Euler
step, so the update is \code{latents + (sigma_i - sigma_{i+1}) * v}.
}
