% tinyrox says don't edit this manually, but it can't stop you!
\name{anvl_ltx_dit}
\alias{anvl_ltx_dit}
\title{LTX-2.3 audio-video DiT transformer (anvl port)}
\description{
anvl re-implementation of \code{diffuseR::ltx23_transformer}
(LTX-2.3 dual-stream audio/video DiT). Video and audio latent tokens
each run a stream of: modulated RMS-normed self-attention (per-head
output gating, split RoPE), text cross-attention (with LTX-2.3
query-side and key/value-side modulation and gating), optional
audio<->video cross-attention (a2v / v2a, with global + per-block
modulation and temporal-only cross RoPE), and a modulated GELU-tanh
feed-forward. Per-token timestep modulation feeds every block; a global
sigma drives the prompt (text) modulation; the output layer applies a
per-token scale/shift from the embedded timestep.
}
\details{
Field names mirror the diffusers / \code{dit_ltx23} module tree so
checkpoint keys map 1:1. Weights travel as a nested named list
(pytree); all 2-D linears are pre-transposed to \code{[in, out]}. The
parameter-free timestep sinusoids and RoPE cos/sin tables are
precomputed host-side (\code{\link{yq_ltx_time_sinusoid}} and the
reference RoPE embedder) and passed as inputs, matching the FLUX.2 /
Z-Image boundary.

}
