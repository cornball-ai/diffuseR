% tinyrox says don't edit this manually, but it can't stop you!
\name{flux2_transformer}
\alias{flux2_transformer}
\title{FLUX.2 transformer model}
\usage{
flux2_transformer(in_channels = 128L, num_layers = 5L, num_single_layers = 20L,
                  attention_head_dim = 128L, num_attention_heads = 24L,
                  joint_attention_dim = 7680L, mlp_ratio = 3,
                  timestep_guidance_channels = 256L,
                  axes_dims_rope = c(32L, 32L, 32L, 32L), rope_theta = 2000,
                  eps = 1e-06, out_channels = NULL)
}
\arguments{
\item{in_channels}{Integer. Packed latent channels (128).}

\item{num_layers}{Integer. Double-stream block count (klein-4B: 5).}

\item{num_single_layers}{Integer. Single-stream block count (20).}

\item{attention_head_dim}{Integer. Per-head dimension.}

\item{num_attention_heads}{Integer. Attention heads.}

\item{joint_attention_dim}{Integer. Text embedding dim (7680).}

\item{mlp_ratio}{Numeric. Feed-forward multiplier (3.0).}

\item{timestep_guidance_channels}{Integer. Sinusoid width (256).}

\item{axes_dims_rope}{Integer vector. Per-axis rotary dims.}

\item{rope_theta}{Numeric. Rotary base frequency (2000).}

\item{eps}{Numeric. Norm epsilon.}

\item{out_channels}{Integer or NULL. Defaults to \code{in_channels}.}
}
\value{
Module whose forward(hidden_states, encoder_hidden_states,
  timestep, image_rotary_emb) returns the predicted velocity for the
  image tokens [B, S_img, out_channels]. \code{timestep} is in sigma
  space (0-1); it is scaled by 1000 internally.
}
\description{
Shared modulation computed once per forward; double blocks over
separate text/image streams, then single (parallel) blocks over the
concatenated [text; image] sequence. Rotary embeddings are
precomputed by the caller with \code{\link{flux_pos_embed}}
(\code{axes_dim = c(32, 32, 32, 32)}, \code{theta = 2000}) over the
concatenated [text; image] 4-axis position ids.
}
