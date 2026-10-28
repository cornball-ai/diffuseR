% tinyrox says don't edit this manually, but it can't stop you!
\name{dit_flux3_modules}
\alias{dit_flux3_modules}
\title{FLUX 3 Transformer Building Blocks}
\description{
Fresh R port of the FLUX 3 blocks from the diffusers reference
implementation (Apache-2.0,
src/diffusers/models/transformers/transformer_flux3.py). Every block
is a parallel attention + SwiGLU feed-forward computed from one
modulated input: per-mode blocks self-attend within one token mode
(image, reference image, or text), joint blocks attend over the
concatenated [text; content] sequence. Modulation is per token,
computed at model level. Module field names mirror the diffusers
state-dict keys 1:1; every linear is bias-free. Reuses
\code{flux2_feed_forward}, \code{flux_apply_rotary_emb} and
\code{.ltx23_sdpa}.
}
