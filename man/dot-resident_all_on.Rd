% tinyrox says don't edit this manually, but it can't stop you!
\name{.resident_all_on}
\alias{.resident_all_on}
\title{TRUE when every staged tensor sits on the expected device type}
\usage{
.resident_all_on(staging, type)
}
\arguments{
\item{staging}{A named list of staging sets.}

\item{type}{"cpu" or "cuda".}
}
\value{
Logical.
}
\description{
TRUE when every staged tensor sits on the expected device type
}
\keyword{internal}
