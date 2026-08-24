% tinyrox says don't edit this manually, but it can't stop you!
\name{st_caps}
\alias{st_caps}
\title{safetensors read-capability probes and upgrade messaging}
\description{
The CRAN safetensors 0.2.1 release reads bfloat16 but cannot write it,
and has no float8 support at all. Both fixes (mlverse/safetensors#11
for bfloat16 write, #13 for float8) merged upstream on 2026-07-31 and
reached CRAN in safetensors 0.3.0 on 2026-08-21, along with #14 for the
>2 GB offset overflow and #10 for empty tensor names.
}
\details{
The gates here stayed runtime probes rather than a version floor, and
that decision is what let the fork requirement retire itself the day
the release landed: nothing had to change for a 0.3.0 user to get the
higher tiers. It also still covers the case a version test cannot --
the fixes existed for three weeks in builds that reported 0.2.1, so
the version number never distinguished them. Probe, do not pin.
Two capabilities matter and they differ:

\itemize{
\item \emph{write} (\code{\link{flux_quantize}}'s internal
\code{.st_can_write}, in quantize_flux.R): needed to BUILD a
quantized artifact in that dtype.
\item \emph{read} (\code{.st_can_read}, here): needed to LOAD a
hosted artifact in that dtype. This is the capability that gates
user-facing recommendations. It is strictly weaker than write:
the CRAN 0.2.1 release reads bfloat16 it cannot write, so the
write probe is the wrong signal for whether a hosted bf16 artifact
will load.
}

Both are capability-probed, never version-pinned. That is why the fork
requirement healed itself when 0.3.0 shipped, and why it will do the
same for whatever lands next.

}
