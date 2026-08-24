% tinyrox says don't edit this manually, but it can't stop you!
\name{.resident_prewarm}
\alias{.resident_prewarm}
\title{Grow the caching allocator's pool in one allocation before a bulk onload}
\usage{
.resident_prewarm(bytes, device, held = NULL)
}
\arguments{
\item{bytes}{Numeric. Host bytes about to be transferred; the pool is
warmed to this plus a small margin for allocator slack.}

\item{device}{Target CUDA device.}

\item{held}{Bytes the caching allocator already holds. NULL measures it.
Pass a value to make the decision deterministic: without CUDA the
measurement is 0, which would always warm, so a test that wants the
skip has to state what the pool holds rather than depend on the
machine having a card. Same reason \code{\link{.resident_check_fits}}
takes \code{free_gb}.}
}
\value{
Invisibly, the bytes requested from the allocator: 0 when the
  pool already covers the transfer and nothing was asked for.
}
\description{
A cold bulk onload grows the pool one \code{cudaMalloc} per tensor, and
the syscalls dominate: SDXL's 8.0 GB pinned set measured 24.16 s on the
first activation against 0.32 s on the second, a 74x ratio, on an
otherwise idle RTX 5060 Ti. One large allocation freed straight back into
the pool lets the transfers carve from cached blocks instead. Same
technique the NF4 LTX loader already uses for its first transformer
onload (83.5 s -> 4.6 s there).
}
\details{
It matters beyond the wall clock: a residency broker with a startup
deadline reads 24 s inside a first activate as a wedged worker.

Best-effort. A card that cannot seat the block in one piece falls back to
the per-tensor path, which is the current behaviour and merely slow, so
the failure is swallowed rather than raised.

}
\keyword{internal}
