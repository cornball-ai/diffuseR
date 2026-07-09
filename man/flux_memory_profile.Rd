% tinyrox says don't edit this manually, but it can't stop you!
\name{flux_memory_profile}
\alias{flux_memory_profile}
\title{Resolve a FLUX memory profile}
\usage{
flux_memory_profile(vram_gb = NULL)
}
\arguments{
\item{vram_gb}{Numeric or NULL. Available VRAM; auto-detected when
NULL (via nvidia-smi).}
}
\value{
List with \code{name}, \code{precision} ("nf4"/"fp8"/"bf16"),
  \code{attn_chunk}, \code{text_device}, \code{phase_offload},
  \code{max_pixels}, and (advisory) \code{fork_suggested} and
  \code{note}.
}
\description{
A thin adapter over \code{\link{recommend}} for the FLUX.1 pipeline,
kept for back-compatibility. \code{recommend("flux1")} is the policy;
this reshapes it into the legacy profile fields the loader consumes.
Precision now rises with VRAM (nf4 default, fp8 GPU-resident on 14 GB+
cards when safetensors can read float8, bf16 on 24 GB+); the old
bands, which put fp8 in a narrow low-VRAM slot it can no longer fit,
were backwards.
}
