% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_tune_gc}
\alias{ltx23_tune_gc}
\title{Tune the torch CUDA allocator for large-model inference}
\usage{
ltx23_tune_gc(footprint_gb, total_gb = NULL)
}
\arguments{
\item{footprint_gb}{Numeric. Expected resident GPU footprint
(weights + activations).}

\item{total_gb}{Numeric or NULL (auto-detect total VRAM).}
}
\value{
Invisibly, the applied reserved rate (NULL if skipped).
}
\description{
By default torch triggers R garbage collection when CUDA allocations
pass ~20\\% of VRAM, which fires on nearly every allocation once a
large model is resident and can slow inference several-fold. Sets the
allocator reserved rate to the actual expected footprint (clamped to
[0.2, 0.92]) and raises the GC call threshold. Must run BEFORE the
first CUDA operation of the session; user-set options are respected.
}
