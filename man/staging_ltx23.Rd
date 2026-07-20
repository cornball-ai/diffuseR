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
Costs: the model's host copies become non-swappable for the life
of the pipeline (no extra RAM - set_data repoints the same
tensors), and page-locking adds ~9s to pipeline load. Measured
post byte-LUT (768x512x49, NF4, RTX 5060 Ti): ~7s saved per render
(warm renders 64-66s pageable vs 57-59s pinned; denoise and decode
identical, the delta is pure transfer), so pinning breaks even on
the second render and costs a single-render session ~2s net. On by
default; page-locking failure falls back silently per component,
and \code{options(diffuseR.pin_staging = FALSE)} before
\code{ltx23_load_pipeline} opts out (e.g. under host memory
pressure, where unswappable pages turn thrashing into OOM).

}
