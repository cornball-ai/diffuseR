% tinyrox says don't edit this manually, but it can't stop you!
\name{.resident_check_fits}
\alias{.resident_check_fits}
\title{Refuse a bulk activation that cannot fit}
\usage{
.resident_check_fits(res, free_gb = NULL, need_bytes = NULL)
}
\arguments{
\item{res}{A resident handle.}

\item{free_gb}{Free VRAM in GB. NULL measures it. Pass a value to
make the decision deterministic: with no GPU the measurement is 0,
which means "cannot tell" and never refuses, so a test that wants
the refusal has to state the budget rather than depend on the
machine having a card.}

\item{need_bytes}{Bytes actually headed for the card. NULL means the
whole pinned set, which is right for a family that onloads everything
and wrong for one that onloads a subset -- SDXL pins 8.0 GB and sends
5.1 GB of it, so charging it the full figure would refuse activations
that fit.}
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
