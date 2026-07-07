% tinyrox says don't edit this manually, but it can't stop you!
\name{flux_memory_profile}
\alias{flux_memory_profile}
\title{Resolve a FLUX memory profile}
\usage{
flux_memory_profile(vram_gb = NULL)
}
\arguments{
\item{vram_gb}{Numeric or NULL. Available VRAM; auto-detected when
NULL (via gpu.ctl or nvidia-smi).}
}
\value{
List with \code{name}, \code{precision} ("nf4"/"fp8"),
  \code{attn_chunk}, \code{text_device}, \code{phase_offload}, and
  \code{max_pixels} (largest validated image area).
}
\description{
Resolve a FLUX memory profile
}
