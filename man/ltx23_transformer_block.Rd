% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_transformer_block}
\alias{ltx23_transformer_block}
\title{LTX-2 transformer block}
\usage{
ltx23_transformer_block(dim, num_attention_heads, attention_head_dim,
                        cross_attention_dim, audio_dim,
                        audio_num_attention_heads, audio_attention_head_dim,
                        audio_cross_attention_dim, video_gated_attn = TRUE,
                        video_cross_attn_adaln = TRUE, audio_gated_attn = TRUE,
                        audio_cross_attn_adaln = TRUE, eps = 1e-06,
                        elementwise_affine = FALSE, rope_type = "split",
                        perturbed_attn = TRUE)
}
\arguments{
\item{cross_attention_dim}{Integer. Text embedding dim for video.}

\item{audio_cross_attention_dim}{Integer. Text embedding dim for audio.}

\item{eps}{Numeric. Norm epsilon.}

\item{elementwise_affine}{Logical. Block norms carry weights (FALSE for LTX).}

\item{rope_type}{"split" or "interleaved".}

\item{perturbed_attn}{Logical. Enable the STG perturbation arguments.}

\item{dim,audio_dim}{Integers. Video/audio stream dimensions.}

\item{num_attention_heads,attention_head_dim}{Video attention shape.}

\item{audio_num_attention_heads,audio_attention_head_dim}{Audio attention shape.}

\item{video_gated_attn,audio_gated_attn}{Logicals. Per-head output gates.}

\item{video_cross_attn_adaln,audio_cross_attn_adaln}{Logicals. LTX-2.3
text cross-attention modulation (9 mod params instead of 6).}
}
\description{
Dual-stream (video + audio) block: modulated self-attention per
modality, text cross-attention per modality (with LTX-2.3 query and
key/value modulation), bidirectional audio-video cross-attention with
global+per-block modulation, and modulated feed-forward.
}
