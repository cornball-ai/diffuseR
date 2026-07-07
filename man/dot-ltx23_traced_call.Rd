% tinyrox says don't edit this manually, but it can't stop you!
\name{.ltx23_traced_call}
\alias{.ltx23_traced_call}
\title{Run a module forward through a shape-specialized trace}
\usage{
.ltx23_traced_call(module, x, forward = NULL, tag = "")
}
\arguments{
\item{module}{The nn_module (identity for the cache key; also the
default callable).}

\item{x}{Input tensor.}

\item{forward}{Optional closure wrapping the call (for extra fixed
arguments like \code{causal}); must be pure in \code{x}.}

\item{tag}{Character. Distinguishes call variants of one module.}
}
\value{
The forward result.
}
\description{
Run a module forward through a shape-specialized trace
}
\keyword{internal}
