% tinyrox says don't edit this manually, but it can't stop you!
\name{.ltx23_staged_onload}
\alias{.ltx23_staged_onload}
\title{Move a pinned component onto the compute device}
\usage{
.ltx23_staged_onload(staging, device)
}
\description{
Non-blocking copies from pinned memory share the default stream,
so later kernels are ordered after them; no explicit sync needed.
}
\keyword{internal}
