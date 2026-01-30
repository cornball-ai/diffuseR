% tinyrox says don't edit this manually, but it can't stop you!
\name{CLIPMLP}
\alias{CLIPMLP}
\title{CLIP MLP Block}
\usage{
CLIPMLP(in_dim, hidden_dim, gelu_type = "tanh")
}
\arguments{
\item{in_dim}{Input dimension}

\item{hidden_dim}{Hidden dimension}

\item{gelu_type}{GELU variant: "tanh" (tanh approximation), "quick" (QuickGELU), "exact" (standard GELU)}
}
\description{
Feed-forward network with configurable activation
}
\keyword{internal}
