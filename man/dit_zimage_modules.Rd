% tinyrox says don't edit this manually, but it can't stop you!
\name{dit_zimage_modules}
\alias{dit_zimage_modules}
\title{Z-Image Transformer Block Modules}
\description{
Fresh R port of the Z-Image DiT building blocks from the diffusers
reference (Apache-2.0,
src/diffusers/models/transformers/transformer_z_image.py). Z-Image is
a single-stream DiT: text and image tokens share one sequence and one
set of block weights. Each block uses sandwich RMSNorms (a learned
norm before AND after both the attention and the feed-forward) and a
scale/gate-only modulation — four chunks (scale_msa, gate_msa,
scale_mlp, gate_mlp), no shift, gates tanh-squashed, scales 1 + x.
The attention is plain joint self-attention, so the FLUX attention
module is reused with bias = FALSE and eps = 1e-5.
}
