% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_transformer}
\alias{ltx23_transformer}
\title{LTX-2.3 video transformer model}
\usage{
ltx23_transformer(in_channels = 128L, out_channels = 128L, patch_size = 1L,
                  patch_size_t = 1L, num_attention_heads = 32L,
                  attention_head_dim = 128L, cross_attention_dim = 4096L,
                  vae_scale_factors = c(8L, 32L, 32L), pos_embed_max_pos = 20L,
                  base_height = 2048L, base_width = 2048L, gated_attn = TRUE,
                  cross_attn_mod = TRUE, audio_in_channels = 128L,
                  audio_out_channels = 128L, audio_patch_size = 1L,
                  audio_patch_size_t = 1L, audio_num_attention_heads = 32L,
                  audio_attention_head_dim = 64L,
                  audio_cross_attention_dim = 2048L, audio_scale_factor = 4L,
                  audio_pos_embed_max_pos = 20L, audio_sampling_rate = 16000L,
                  audio_hop_length = 160L, audio_gated_attn = TRUE,
                  audio_cross_attn_mod = TRUE, num_layers = 48L,
                  norm_eps = 1e-06, rope_theta = 10000,
                  rope_double_precision = TRUE, causal_offset = 1L,
                  timestep_scale_multiplier = 1000,
                  cross_attn_timestep_scale_multiplier = 1000,
                  rope_type = "split", perturbed_attn = TRUE)
}
\arguments{
\item{cross_attention_dim}{Integer. Video text embedding dimension.}

\item{vae_scale_factors}{Integer vector. VAE (time, height, width) scales.}

\item{audio_cross_attention_dim}{Integer. Audio text embedding dimension.}

\item{num_layers}{Integer. Transformer block count.}

\item{norm_eps}{Numeric. Norm epsilon.}

\item{rope_type}{"split" (LTX-2.3) or "interleaved".}

\item{in_channels,out_channels}{Integers. Video latent channels.}

\item{patch_size,patch_size_t}{Integers. Video patch sizes.}

\item{num_attention_heads,attention_head_dim}{Video attention shape.}

\item{pos_embed_max_pos,base_height,base_width}{RoPE base grid.}

\item{audio_in_channels,audio_out_channels}{Integers. Audio latent channels.}

\item{audio_patch_size,audio_patch_size_t}{Integers. Audio patch sizes.}

\item{audio_num_attention_heads,audio_attention_head_dim}{Audio attention shape.}

\item{audio_scale_factor,audio_pos_embed_max_pos,audio_sampling_rate,audio_hop_length}{Audio latent grid parameters.}

\item{rope_theta,rope_double_precision,causal_offset}{RoPE parameters.}

\item{timestep_scale_multiplier,cross_attn_timestep_scale_multiplier}{Timestep scaling (inputs arrive already scaled; the ratio modulates
the a2v/v2a gates).}

\item{gated_attn,cross_attn_mod,audio_gated_attn,audio_cross_attn_mod,perturbed_attn}{LTX-2.3 feature flags (all TRUE for the 2.3 checkpoints).}
}
\description{
Dual-stream audio/video DiT. Text embeddings arrive already projected
to the video (\code{inner_dim}) and audio (\code{audio_inner_dim})
dimensions by the connector modules.
}
