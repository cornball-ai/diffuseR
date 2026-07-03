% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_snake_beta}
\alias{ltx23_snake_beta}
\title{SnakeBeta activation}
\usage{
ltx23_snake_beta(channels, eps = 1e-09)
}
\arguments{
\item{channels}{Integer.}

\item{eps}{Numeric.}
}
\description{
\code{x + (1 / (beta + eps)) * sin(x * alpha)^2} with per-channel
log-scale alpha/beta parameters.
}
