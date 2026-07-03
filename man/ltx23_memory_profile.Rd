% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_memory_profile}
\alias{ltx23_memory_profile}
\title{Get an LTX-2.3 memory profile}
\usage{
ltx23_memory_profile(vram_gb = NULL)
}
\arguments{
\item{vram_gb}{Numeric or NULL (auto-detect free VRAM).}
}
\value{
Named list with device/dtype placement, \code{attn_chunk},
  \code{pin_weights}, and resolution caps.
}
\description{
Selects component placement and attention chunking for the available
VRAM. The fp8 transformer weights (~21 GB) always stay CPU-resident
and stream per-linear; profiles differ in what else lives on the GPU
and how hard the attention is chunked.
}
