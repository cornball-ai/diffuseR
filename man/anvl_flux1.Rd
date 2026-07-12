% tinyrox says don't edit this manually, but it can't stop you!
\name{anvl_flux1}
\alias{anvl_flux1}
\title{FLUX.1 MMDiT transformer (anvl port)}
\description{
anvl re-implementation of \code{diffuseR::flux_transformer}
(FluxTransformer2DModel, FLUX.1-schnell). Same double-stream MMDiT +
single-stream family as the FLUX.2 port
(\code{\link{yq_flux2_transformer}}) and reuses its helpers, but with
three deliberate differences from FLUX.2:
\enumerate{
\item \strong{Per-block modulation (adaLN-Zero).} FLUX.2 computes
the (shift, scale, gate) modulation once and shares it across
every block; FLUX.1 computes it inside each block from
\code{silu(temb)} via that block's own \code{norm1}/
\code{norm1_context}/\code{norm} linear.
\item \strong{No guidance embedding (schnell is CFG-free).} The
timestep conditioning is \code{timestep_embedder(sinusoid) +
text_embedder(pooled CLIP)} with no guidance term.
\item \strong{Non-fused single-stream block, GELU-tanh FFN.} FLUX.1
keeps separate \code{to_q/to_k/to_v} + \code{proj_mlp} and a
\code{proj_out} over \code{cat(attn, mlp)} (FLUX.2 fuses
QKV+MLP), and its feed-forward is GELU-tanh (FLUX.2 is SwiGLU).
Every linear carries a bias (FLUX.2's are bias-free).
}
Weights travel as a nested named list (pytree); see
\code{\link{yq_flux1_load_weights}}. The timestep sinusoid
(\code{\link{yq_flux1_time_embed}}) and RoPE cos/sin tables are
precomputed host-side and passed as inputs, matching the diffusers
pipeline boundary and the FLUX.2 port.
}
