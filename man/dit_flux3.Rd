% tinyrox says don't edit this manually, but it can't stop you!
\name{dit_flux3}
\alias{dit_flux3}
\title{FLUX 3 Transformer}
\description{
Fresh R port of Flux3Transformer2DModel from the diffusers reference
implementation (Apache-2.0,
src/diffusers/models/transformers/transformer_flux3.py). Defaults are
the FLUX 3 Image Klein configuration: 5 per-mode blocks for each of
the image, reference-image and text modes, then 28 joint blocks.
Conditioning is a per-token timestep embedding (the image tokens at
the flow timestep, text and reference tokens at 0); the step-distilled
(flow-map) checkpoint adds the embedding of the timestep each step
jumps to.
}
