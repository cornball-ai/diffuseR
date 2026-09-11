% tinyrox says don't edit this manually, but it can't stop you!
\name{.staged_on}
\alias{.staged_on}
\title{Is every pinned tensor of a component on this device?}
\usage{
.staged_on(staging, device)
}
\arguments{
\item{staging}{A component's staging set: the list of
\code{list(live, pinned)} pairs \code{.pin_component} returned.}

\item{device}{The compute device, as a string (\code{"cuda"},
\code{"cuda:0"}) or a \code{torch_device}; only its type is compared.}
}
\value{
TRUE when every pair's live tensor is on the device's type;
  FALSE on any mismatch or unreadable pair. Vacuously TRUE for an
  empty staging set, which holds nothing to move.
}
\description{
The check a caller makes before skipping an onload. It asks EVERY
pair, not the first one: a component is on the card when all of it
is, and a probe of one tensor cannot tell a resident component from
one whose onload failed partway. That partial state is real -- an
onload that runs out of device memory leaves the pairs it copied on
the card and the rest on the host -- and a first-pair probe reports
it as "already resident", so every later phase skips the onload and
dies on a device mismatch, on every call, until the process ends.
That is how a gpuhost's LTX entry wedged for a whole show on
2026-09-10: one failed encoder onload, then "mat2 is on cpu" from
every request after it.
}
\keyword{internal}
