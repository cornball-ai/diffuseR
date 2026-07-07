% tinyrox says don't edit this manually, but it can't stop you!
\name{rope_ltx23}
\alias{rope_ltx23}
\title{LTX-2.3 Rotary Positional Embeddings}
\description{
Fresh R port of the LTX rotary positional embedding scheme from the
diffusers reference implementation (Apache-2.0,
src/diffusers/models/transformers/transformer_ltx2.py). LTX 2.3 uses
the "split" RoPE layout everywhere; "interleaved" is kept for
completeness. Frequencies are computed in float64 per the checkpoint
config (frequencies_precision) and applied in float32.
}
