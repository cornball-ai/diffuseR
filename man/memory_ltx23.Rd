% tinyrox says don't edit this manually, but it can't stop you!
\name{memory_ltx23}
\alias{memory_ltx23}
\title{LTX-2.3 Memory Profiles and CUDA GC Tuning}
\description{
Memory management for running the 22B LTX-2.3 transformer on limited
VRAM, built on the patterns proven in the whisper and chatterbox
packages: torch allocator GC tuning before the first CUDA op, fp8
CPU-resident streaming weights, query-chunked attention, and
phase-sequential component placement.
}
