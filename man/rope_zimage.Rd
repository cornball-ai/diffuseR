% tinyrox says don't edit this manually, but it can't stop you!
\name{rope_zimage}
\alias{rope_zimage}
\title{Z-Image Rotary Positional Embeddings and Patchify Helpers}
\description{
Fresh R port of the Z-Image position scheme from the diffusers
reference (Apache-2.0,
src/diffusers/models/transformers/transformer_z_image.py RopeEmbedder,
create_coordinate_grid, _patchify_image, _pad_with_ids, unpatchify).
Z-Image uses 3-axis interleaved RoPE with theta 256; frequencies are
built in float64 but the angles are cast to float32 before cos/sin
(torch.polar on a .float() tensor), which differs measurably from the
FLUX convention at large positions. Every sub-sequence is padded to a
multiple of 32 (SEQ_MULTI_OF); caption positions are a 1-based ramp on
axis 1 built over the padded length, image positions sit on axes 2/3
with axis 1 offset just past the caption.
}
