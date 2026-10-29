% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_memory_profile}
\alias{ltx23_memory_profile}
\title{Get an LTX-2.3 memory profile}
\usage{
ltx23_memory_profile(vram_gb = NULL, host_ram_gb = NULL)
}
\arguments{
\item{vram_gb}{Numeric or NULL (auto-detect free VRAM).}

\item{host_ram_gb}{Numeric or NULL (auto-detect available RAM; NA
when undetectable, which keeps the all-in-RAM defaults).}
}
\value{
Named list with device/dtype placement, \code{attn_chunk},
  \code{pin_weights}, \code{resident_gb}, \code{stream_text},
  \code{split_text}, \code{resident_from}, and resolution caps.
}
\description{
Selects transformer precision, component placement, and attention
chunking for the available VRAM. Measured on an RTX 5060 Ti (16 GB):
fp8 streaming peaks ~11.6 GB (without phase offloading) at
512x320x49; NF4 keeps the whole 22B transformer resident (~12.5 GB)
and removes the ~21 GB/step PCIe weight streaming. The NF4 profile
renders 1280x704x121 with audio in ~23 min at a 15.7 GB peak
(tiled VAE decode, in-place feed-forward GELU, and the default
\code{diffuseR.attn_budget} of 1.5e8 all required at that size).
}
\details{
Below 14 GB the NF4 transformer keeps \code{resident_gb} on the GPU
and streams its remaining blocks from pinned host memory every step
(pass it to \code{\link{ltx23_load_pipeline}}), and below 12 GB the
Gemma 4 encoder streams its layers (\code{stream_text}; pass it as
\code{stream} to \code{\link{load_gemma4_text_encoder}}). Measured
with the rest of a 16 GB card held back: LTX-2.5 768x512x121 renders
in ~84 s on both 12 GB (8 GB resident) and 8 GB (4 GB resident), the
same as on the full card.

\describe{
\item{precision "nf4"}{Weights resident on the GPU (or partly
streamed); fastest steps; about 9 percent weight round-trip error.}
\item{precision "fp8"}{Weights CPU-resident, streamed per linear;
near-bf16 quality; each step pays the PCIe transfer.}
}

Host RAM decides two more fields for the GPU tiers. \code{split_text}
asks the caller to encode in its own phase (Gemma, then the
connectors, each freed before the next loads; see
\code{\link{ltx23_text_embeds}}) instead of holding the encoder next
to the render pipeline. \code{resident_from} is passed to
\code{\link{ltx23_load_pipeline}}: \code{"disk"} drops the host copy
of the GPU-resident transformer blocks. Below 32 GB available the
text phase splits off; below 22 GB the resident blocks come from
disk (measured on LTX-2.5 at 768x512x121 with an 8 GB card: the
render phase peaks at 20.1 GB of host RAM with host copies and 16.0
GB without, for ~8 s more per render).

}
