% tinyrox says don't edit this manually, but it can't stop you!
\name{.resident_check_fits}
\alias{.resident_check_fits}
\title{Refuse a bulk activation that cannot fit}
\usage{
.resident_check_fits(res)
}
\arguments{
\item{res}{A resident handle.}
}
\value{
Invisibly TRUE, or an error naming both figures.
}
\description{
Fails before the transfer rather than part-way through it. A partial
onload that OOMs is recoverable (activation rolls back), but it wastes
the transfer and reports a libtorch allocator error instead of the
actual problem, which is that this model does not fit this card.
}
\keyword{internal}
