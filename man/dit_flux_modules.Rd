% tinyrox says don't edit this manually, but it can't stop you!
\name{dit_flux_modules}
\alias{dit_flux_modules}
\title{FLUX Transformer Building Blocks}
\description{
Fresh R port of the FLUX MMDiT blocks from the diffusers reference
implementation (Apache-2.0,
src/diffusers/models/transformers/transformer_flux.py and
src/diffusers/models/normalization.py). Module field names mirror the
diffusers state-dict keys 1:1 so checkpoints load without remapping.
Reuses the LTX primitives \code{ltx23_rms_norm}, \code{.ltx23_sdpa}
and \code{ltx23_feed_forward}.
}
