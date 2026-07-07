% tinyrox says don't edit this manually, but it can't stop you!
\name{rope_flux}
\alias{rope_flux}
\title{FLUX Rotary Positional Embeddings}
\description{
Fresh R port of the FLUX rotary positional embedding scheme from the
diffusers reference implementation (Apache-2.0,
src/diffusers/models/transformers/transformer_flux.py FluxPosEmbed and
src/diffusers/models/embeddings.py get_1d_rotary_pos_embed /
apply_rotary_emb). FLUX uses the interleaved adjacent-pair convention
(use_real_unbind_dim = -1) with per-axis frequencies computed in
float64 and applied in float32. Text tokens carry all-zero ids, so
they receive the identity rotation.
}
