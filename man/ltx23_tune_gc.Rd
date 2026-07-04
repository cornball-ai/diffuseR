% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_tune_gc}
\alias{ltx23_tune_gc}
\title{Tune the torch CUDA allocator for large-resident inference}
\usage{
ltx23_tune_gc(allocated_rate = 0.7)
}
\arguments{
\item{allocated_rate}{Numeric. Allocated/total ratio above which the
allocator requests a full R collection (torch default 0.8).}
}
\value{
Invisibly, NULL.
}
\description{
Configures the torch CUDA allocator for pipelines that keep most of
the GPU occupied by resident weights (cf. the mlverse torch
memory-management article). \code{torch.cuda_allocator_reserved_rate}
is deliberately left at its default: raising it (a common recipe for
small-resident models) disables R garbage collection while reserved
memory is below the rate, which starves garbage-heavy inference. This
instead lowers \code{torch.cuda_allocator_allocated_rate} so full
collections engage earlier under pressure, and defaults
\code{PYTORCH_CUDA_ALLOC_CONF} to expandable segments (must run
before the first CUDA allocation). User-set options are respected.
}
