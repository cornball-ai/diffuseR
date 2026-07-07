% tinyrox says don't edit this manually, but it can't stop you!
\name{dit_flux2}
\alias{dit_flux2}
\title{FLUX.2 Transformer (MMDiT)}
\description{
Fresh R port of Flux2Transformer2DModel from the diffusers reference
implementation (Apache-2.0,
src/diffusers/models/transformers/transformer_flux2.py). Defaults are
the klein-4B configuration (5 double + 20 single blocks). Guidance
embeddings (FLUX.2-dev) are not implemented; klein is step-distilled
with \code{guidance_embeds = false}. Timestep conditioning has no
pooled-text component, and the three modulation projections are
shared across all blocks.
}
