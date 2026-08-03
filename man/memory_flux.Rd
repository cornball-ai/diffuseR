% tinyrox says don't edit this manually, but it can't stop you!
\name{memory_flux}
\alias{memory_flux}
\title{FLUX Memory Profiles}
\description{
VRAM-based execution profiles for the FLUX.1-schnell pipeline. The
12B transformer runs NF4 (~7 GB) or fp8 (~12 GB), phase-onloaded to
the GPU for denoise; the T5-XXL text encoder phase-onloads to the GPU
(bfloat16, pinned) on 14 GB+ cards and computes on the CPU (float32)
below that, where its ~9.8 GB encode phase does not fit.
}
