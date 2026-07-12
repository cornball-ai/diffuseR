% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_sd21_ddim_step}
\alias{yq_sd21_ddim_step}
\title{One v-prediction DDIM step (anvl)}
\usage{
yq_sd21_ddim_step(sample, model_output, coeff, device = "cpu")
}
\arguments{
\item{sample}{AnvlArray \code{[B, 4, H, W]} current noisy latents.}

\item{model_output}{AnvlArray \code{[B, 4, H, W]} predicted velocity
(post-CFG).}

\item{coeff}{One element of \code{yq_sd21_ddim_sigmas()$coeff}.}

\item{device}{Character. Device for the scalar coefficients.}
}
\value{
AnvlArray \code{[B, 4, H, W]} previous (less noisy) sample.
}
\description{
Applies a single deterministic (eta 0) v-prediction DDIM update using
the host-side coefficients from \code{\link{yq_sd21_ddim_sigmas}}:
\deqn{x_0 = a_t \cdot sample - b_t \cdot v}
\deqn{\epsilon = a_t \cdot v + b_t \cdot sample}
\deqn{prev = a_{prev} \cdot x_0 + d_{prev} \cdot \epsilon}
All four coefficients enter as f32 scalars (auto-broadcast), so the
latents never leave the device.
}
