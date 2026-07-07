% tinyrox says don't edit this manually, but it can't stop you!
\name{dit_flux}
\alias{dit_flux}
\title{FLUX Transformer (MMDiT)}
\description{
Fresh R port of FluxTransformer2DModel from the diffusers reference
implementation (Apache-2.0,
src/diffusers/models/transformers/transformer_flux.py). The module
tree mirrors the diffusers state-dict keys 1:1, so checkpoints load
without remapping. FLUX.1-schnell has no guidance embedder
(guidance_embeds = FALSE); the guidance-distilled dev variant is not
implemented.
}
