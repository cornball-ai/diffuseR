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
\keyword{internal}
