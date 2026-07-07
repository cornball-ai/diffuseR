% tinyrox says don't edit this manually, but it can't stop you!
\name{staging_ltx23}
\alias{staging_ltx23}
\title{Pinned Staging for Phase-Sequential Components}
\description{
Phase offloading moves each large component (transformer,
connectors, VAEs, vocoder) between CPU and GPU every render. From
pageable memory those copies run through the driver's bounce
buffer at a fraction of PCIe speed; page-locked (pinned) host
memory transfers by DMA at full rate. Each component's parameters
and buffers are pinned once at load; onload swaps every tensor to
a non-blocking GPU copy of its pinned source, and offload simply
re-points the tensors at the still-valid pinned copies — weights
are immutable during inference, so offload moves no bytes at all.
}
\details{
Costs: the pinned copies are non-swappable host RAM for the life
of the pipeline (about the model's CPU footprint), and page-locking
~20GB adds ~30s to pipeline load. Measured per render the win is
small (~2.7s: warm pinned onload 0.54s vs pageable 0.99s, offload
0.06s vs 2.36s) because the real phase-transition cost was
allocator pool regrowth, fixed separately by the loader's pool
pre-warm and by not emptying the CUDA cache between phases. Off by
default; enable for long multi-render sessions with
\code{options(diffuseR.pin_staging = TRUE)} before
\code{ltx23_load_pipeline} (breaks even after ~11 renders).

}
