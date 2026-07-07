% tinyrox says don't edit this manually, but it can't stop you!
\name{flux2_modulation}
\alias{flux2_modulation}
\title{FLUX.2 shared modulation projection}
\usage{
flux2_modulation(dim, mod_param_sets = 2L, bias = FALSE)
}
\arguments{
\item{dim}{Integer. Model dimension.}

\item{mod_param_sets}{Integer. Number of (shift, scale, gate) triples.}

\item{bias}{Logical.}
}
\description{
\code{linear(silu(temb))} producing \code{mod_param_sets} triples of
(shift, scale, gate). Computed once per forward at model level and
broadcast to every block. Reference: Flux2Modulation.
}
