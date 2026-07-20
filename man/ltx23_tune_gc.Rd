% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_tune_gc}
\alias{ltx23_tune_gc}
\title{Tune the torch CUDA allocator for large-resident inference}
\usage{
ltx23_tune_gc(footprint_gb = 12, total_gb = NULL)
}
\arguments{
\item{footprint_gb}{Numeric. Expected resident GPU footprint in GB
(NF4 transformer: ~12).}

\item{total_gb}{Numeric or NULL (auto-detect total VRAM).}
}
\value{
Invisibly, the applied reserved rate (NULL if skipped).
}
\description{
Stops the allocator GC storm (cf. ~/skills/torch
torch-jit-gc-performance.md): lantern proactively calls R's gc()
whenever reserved memory exceeds \code{torch.cuda_allocator_reserved_rate}
(default 0.20) of the card. With ~75\\% of VRAM occupied by resident
weights that fires on nearly every allocation. Raising the rate to the
actual footprint is safe here because the LTX hot loops compute into
persistent scratch buffers (near-zero per-step garbage). Also raises
the host-allocation GC threshold and defaults
\code{PYTORCH_CUDA_ALLOC_CONF} to expandable segments. User-set
options win.
}
\details{
\code{start_torch()} reads the gate options exactly once, so setting
them after torch has started is inert on its own. The three CUDA
gates are therefore also pushed into the live allocator here (the
\code{.flux_gc_gates} pattern), which makes this function effective
whenever it runs. The host-side \code{torch.threshold_call_gc} has
no live setter; the package defaults it in \code{.onLoad} so torch
reads it at init in any session that loads diffuseR before running
torch ops.

}
