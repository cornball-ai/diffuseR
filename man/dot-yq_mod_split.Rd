% tinyrox says don't edit this manually, but it can't stop you!
\name{.yq_mod_split}
\alias{.yq_mod_split}
\title{Split a shared modulation tensor into (shift, scale, gate) triples}
\usage{
.yq_mod_split(mod, n_sets, dim)
}
\arguments{
\item{mod}{AnvlArray \code{[N, 3 * n_sets * dim]}.}

\item{n_sets}{Integer. Number of (shift, scale, gate) triples.}

\item{dim}{Integer. Model dimension.}
}
\value{
List of \code{n_sets} lists, each \code{list(shift, scale, gate)}.
}
\description{
Mirrors diffusers \code{Flux2Modulation.split}: a modulation
projection outputs \code{[N, 3 * n_sets * dim]}; this unsqueezes to
\code{[N, 1, ...]} and slices it into \code{n_sets} lists of three
\code{[N, 1, dim]} tensors.
}
\keyword{internal}
