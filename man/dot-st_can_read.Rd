% tinyrox says don't edit this manually, but it can't stop you!
\name{.st_can_read}
\alias{.st_can_read}
\title{Probe whether the installed safetensors can READ a dtype}
\usage{
.st_can_read(dtype = c("bfloat16", "float8_e4m3fn"))
}
\arguments{
\item{dtype}{"bfloat16" or "float8_e4m3fn".}
}
\value{
Logical.
}
\description{
Hand-builds a tiny safetensors file of the dtype (via
\code{.st_write_min}, no safetensors writer involved) and tries to
load it back. Cached per session;
\code{options(diffuseR.st_read_caps = list(bfloat16 = TRUE, ...))}
overrides the probe for tests and for forcing a tier.
}
\keyword{internal}
