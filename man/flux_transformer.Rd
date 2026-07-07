% tinyrox says don't edit this manually, but it can't stop you!
\name{flux_transformer}
\alias{flux_transformer}
\title{FLUX transformer model}
\usage{
flux_transformer(in_channels = 64L, num_layers = 19L, num_single_layers = 38L,
                 attention_head_dim = 128L, num_attention_heads = 24L,
                 joint_attention_dim = 4096L, pooled_projection_dim = 768L,
                 axes_dims_rope = c(16L, 56L, 56L), out_channels = NULL)
}
\arguments{
\item{in_channels}{Integer. Packed latent channels (64).}

\item{num_layers}{Integer. Double-stream block count.}

\item{num_single_layers}{Integer. Single-stream block count.}

\item{attention_head_dim}{Integer. Per-head dimension.}

\item{num_attention_heads}{Integer. Attention heads.}

\item{joint_attention_dim}{Integer. T5 embedding dim (4096).}

\item{pooled_projection_dim}{Integer. CLIP pooled dim (768).}

\item{axes_dims_rope}{Integer vector. Per-axis rotary dims.}

\item{out_channels}{Integer or NULL. Output channels (defaults to
\code{in_channels}).}
}
\value{
Module whose forward(hidden_states, encoder_hidden_states,
  pooled_projections, timestep, image_rotary_emb) returns the
  predicted velocity for the image tokens [B, S_img, out_channels].
  \code{timestep} is in sigma space (0-1); it is scaled by 1000
  internally, matching the reference.
}
\description{
19 double-stream (MMDiT) blocks followed by 38 single-stream blocks
over the joint [text; image] sequence, with adaLN-Zero conditioning on
timestep + pooled CLIP text. Rotary embeddings are precomputed by the
caller with \code{flux_pos_embed} (they are static across denoise
steps). Defaults are the FLUX.1-schnell configuration.
}
