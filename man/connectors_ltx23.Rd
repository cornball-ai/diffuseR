% tinyrox says don't edit this manually, but it can't stop you!
\name{connectors_ltx23}
\alias{connectors_ltx23}
\title{LTX-2.3 Text Embedding Connectors}
\description{
Fresh R port of the LTX text connectors from the diffusers reference
(Apache-2.0, src/diffusers/pipelines/ltx2/connectors.py). The
connectors take raw stacked per-layer Gemma3 hidden states
[batch, seq, caption_channels, num_layers + 1], normalize and project
them per modality, replace padding with learnable registers, and run a
small 1D transformer per modality to produce the DiT text embeddings.
}
