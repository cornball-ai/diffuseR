% tinyrox says don't edit this manually, but it can't stop you!
\name{st_caps}
\alias{st_caps}
\title{safetensors read-capability probes and fork messaging}
\description{
CRAN safetensors (<= 0.2.1) reads bfloat16 but cannot write it, and
has no float8 support at all; the fixes are upstream
(mlverse/safetensors#11 for bfloat16 write, #13 for float8) and in the
cornball-ai/safetensors fork. Two capabilities matter and they differ:
}
\details{
\itemize{
\item \emph{write} (\code{\link{flux_quantize}}'s internal
\code{.st_can_write}, in quantize_flux.R): needed to BUILD a
quantized artifact in that dtype.
\item \emph{read} (\code{.st_can_read}, here): needed to LOAD a
hosted artifact in that dtype. This is the capability that gates
user-facing recommendations. It is strictly weaker than write:
CRAN safetensors reads bfloat16 it cannot write, so the write
probe is the wrong signal for whether a hosted bf16 artifact will
load.
}

Both are capability-probed, never version-pinned, so the fork
requirement self-heals the day the fixes reach CRAN.

}
