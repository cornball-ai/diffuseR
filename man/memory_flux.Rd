% tinyrox says don't edit this manually, but it can't stop you!
\name{memory_flux}
\alias{memory_flux}
\title{FLUX Memory Profiles}
\description{
VRAM-based execution profiles for the FLUX.1-schnell pipeline. The
12B transformer runs NF4 (~7 GB) or fp8 (~12 GB), both GPU-resident;
the T5-XXL text encoder runs float32 on the CPU by default.
}
