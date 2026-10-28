% tinyrox says don't edit this manually, but it can't stop you!
\name{flux3_schedule}
\alias{flux3_schedule}
\title{FLUX 3 distilled sampling schedule}
\usage{
flux3_schedule(num_inference_steps = 16L, shift = 6.93, device = "cpu")
}
\arguments{
\item{num_inference_steps}{Integer (16 for the distilled checkpoint).}

\item{shift}{Numeric. Static timestep shift.}

\item{device}{Character. Device.}
}
\value{
A FlowMatch schedule from \code{\link{flowmatch_set_timesteps}}.
}
\description{
FlowMatch Euler with the static shift of the FLUX 3 recipe (6.93)
over \code{sigmas = linspace(1, 1 / n, n)}. Reference:
Flux3SetTimestepsStep.
}
