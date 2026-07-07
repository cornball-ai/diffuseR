% tinyrox says don't edit this manually, but it can't stop you!
\name{dit_flux2_modules}
\alias{dit_flux2_modules}
\title{FLUX.2 Transformer Building Blocks}
\description{
Fresh R port of the FLUX.2 MMDiT blocks from the diffusers reference
implementation (Apache-2.0,
src/diffusers/models/transformers/transformer_flux2.py). Key
differences from FLUX.1: modulation is computed ONCE at model level
by shared \code{flux2_modulation} projections and passed into the
blocks (block norms are parameterless), feed-forwards use SwiGLU with
the gate fused into \code{linear_in}, the single-stream block is a
ViT-22B-style parallel block with fully fused projections, and every
linear is bias-free. Module field names mirror the diffusers
state-dict keys 1:1. Reuses \code{flux_attention} (bias = FALSE),
\code{ltx23_rms_norm}, \code{.ltx23_sdpa}, and
\code{flux_apply_rotary_emb}.
}
