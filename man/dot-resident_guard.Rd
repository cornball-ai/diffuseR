% tinyrox says don't edit this manually, but it can't stop you!
\name{.resident_guard}
\alias{.resident_guard}
\title{Refuse operations that the current state cannot serve}
\usage{
.resident_guard(res, verb)
}
\arguments{
\item{res}{A resident handle.}

\item{verb}{What the caller is attempting, for the message.}
}
\value{
Invisibly TRUE, or an error.
}
\description{
Refuse operations that the current state cannot serve
}
\keyword{internal}
