% tinyrox says don't edit this manually, but it can't stop you!
\name{.staged_onload}
\alias{.staged_onload}
\title{Move a pinned component onto the compute device}
\usage{
.staged_onload(staging, device)
}
\description{
Non-blocking copies from pinned memory share the default stream,
so later kernels are ordered after them; no explicit sync needed.
}
\details{
Idempotent PER PAIR: a tensor already on the device is left where it
is, so a resident component costs nothing to onload again (no
re-transfer of weights over themselves, which fragments the
allocator pool) and a component whose earlier onload stopped partway
is completed rather than restarted.

}
\keyword{internal}
