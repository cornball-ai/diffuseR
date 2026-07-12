% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_sdxl_ddim_step}
\alias{yq_sdxl_ddim_step}
\title{One epsilon-prediction DDIM step (anvl)}
\usage{
yq_sdxl_ddim_step(sample, model_output, coeff, device = "cpu")
}
\arguments{
\item{sample}{AnvlArray \code{[B, 4, H, W]} current noisy latents.}

\item{model_output}{AnvlArray \code{[B, 4, H, W]} predicted noise
(post-CFG).}

\item{coeff}{One element of \code{yq_sdxl_ddim_schedule()$coeff}.}

\item{device}{Character. Device for the scalar coefficients.}
}
\value{
AnvlArray \code{[B, 4, H, W]} previous (less noisy) sample.
}
\description{
Applies a single deterministic (eta 0) epsilon DDIM update using the
host-side coefficients from \code{\link{yq_sdxl_ddim_schedule}}:
\deqn{x_0 = (sample - b_t \cdot \epsilon) / a_t}
\deqn{prev = a_{prev} \cdot x_0 + d_{prev} \cdot \epsilon}
where \eqn{\epsilon} is the (post-CFG) predicted noise. The reciprocal
\code{1 / a_t} is precomputed host-side so the update is scalar-broadcast
multiplies and the latents never leave the device.
}
