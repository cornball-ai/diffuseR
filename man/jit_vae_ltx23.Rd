% tinyrox says don't edit this manually, but it can't stop you!
\name{jit_vae_ltx23}
\alias{jit_vae_ltx23}
\title{JIT-Traced Decode for the LTX-2.3 VAEs and Vocoder}
\description{
The video/audio decoders and the vocoder are static feed-forward
graphs, so \code{torch::jit_trace} converts them wholesale: one
R-to-libtorch crossing per forward, intermediates freed eagerly by
libtorch instead of accumulating as R handles until gc. Traces are
shape-specialized (runtime sizes bake into the graph as constants;
a mismatched input errors), so they are cached per instance, input
shape, dtype, device, and call tag, and re-traced on a miss.
}
\details{
A trace captures the module's weight tensors, which would pin them
on the GPU across phase offloads; the pipeline releases all traces
whenever a component offloads (\code{.ltx23_release_vae_traces}).

Tracing hazard on this lantern build: if the allocator callback
runs R's gc \emph{during} trace recording (memory pressure), the
recorded graph can capture garbage argument values (observed as
corrupted \code{narrow} starts on the full-size decoder; verified
5/5 clean once gc cannot fire mid-trace). Defenses, in order: a
gc + cache flush right before each trace so pressure starts near
zero, \code{tryCatch} around trace and replay, and a one-time
validation of every fresh trace against the eager output — any
mismatch permanently blacklists that shape and runs eager. With
those in place the traced path cannot corrupt output — but per
render it measured slower than eager (traces are released on phase
offload, so every render re-pays trace + validation), so it stays
opt-in: \code{options(diffuseR.jit_vae = TRUE)}.

}
