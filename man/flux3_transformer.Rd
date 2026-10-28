% tinyrox says don't edit this manually, but it can't stop you!
\name{flux3_transformer}
\alias{flux3_transformer}
\title{FLUX 3 transformer model}
\usage{
flux3_transformer(
  in_channels = 128L,
  cond_in_channels = 128L,
  num_layers = 5L,
  num_single_layers = 28L,
  attention_head_dim = 128L,
  num_attention_heads = 24L,
  joint_attention_dim = 20480L,
  timestep_channels = 256L,
  mlp_ratio = 3,
  axes_dims_rope = c(32L, 32L, 32L, 32L),
  rope_theta = 10000,
  eps = 1e-06,
  flow_map = FALSE
)
}
\arguments{
\item{in_channels}{Integer. Packed latent channels (128).}

\item{cond_in_channels}{Integer. Reference-image token channels (128).}

\item{num_layers}{Integer. Per-mode blocks per mode (5).}

\item{num_single_layers}{Integer. Joint blocks (28).}

\item{attention_head_dim}{Integer. Per-head dimension (128).}

\item{num_attention_heads}{Integer. Attention heads (24).}

\item{joint_attention_dim}{Integer. Text embedding dim (20480).}

\item{timestep_channels}{Integer. Sinusoid width (256).}

\item{mlp_ratio}{Numeric. Feed-forward multiplier (3.0).}

\item{axes_dims_rope}{Integer vector. Per-axis rotary dims.}

\item{rope_theta}{Numeric. Rotary base frequency (10000).}

\item{eps}{Numeric. Norm epsilon.}

\item{flow_map}{Logical. Step-distilled checkpoint: adds
\code{target_timestep_embedder} and requires \code{target_timestep}.}
}
\value{
Module whose forward(hidden_states, encoder_hidden_states,
  timestep, img_ids, txt_ids, image_cond_hidden_states, image_cond_ids,
  target_timestep, chunk_size) returns the predicted velocity of the
  image tokens [B, S_img, in_channels]. Timesteps are flow times in
  [0, 1]; they are scaled by 1000 internally.
}
\description{
The forward mirrors the reference: per-mode blocks for the text,
image and (optional) reference tokens, each with its own rotary
embedding, then joint blocks over [text; image; reference] and an
output projection of the image tokens only. Modulation is computed
once per mode on the [B, 1, dim] embedding and broadcast over the
mode's tokens (the reference expands to [B, S, dim] first; every row
is identical).
}
