% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx2_text_connectors}
\alias{ltx2_text_connectors}
\title{LTX2 Text Connectors}
\usage{
ltx2_text_connectors(
  caption_channels = 3840L,
  text_proj_in_factor = 49L,
  video_connector_num_attention_heads = 30L,
  video_connector_attention_head_dim = 128L,
  video_connector_num_layers = 2L,
  video_connector_num_learnable_registers = NULL,
  audio_connector_num_attention_heads = 30L,
  audio_connector_attention_head_dim = 128L,
  audio_connector_num_layers = 2L,
  audio_connector_num_learnable_registers = NULL,
  connector_rope_base_seq_len = 4096L,
  rope_theta = 10000,
  rope_double_precision = TRUE,
  causal_temporal_positioning = FALSE,
  rope_type = "split"
)
}
\arguments{
\item{caption_channels}{Integer. Dimension of caption embeddings (default 3840).}

\item{text_proj_in_factor}{Integer. Factor for input projection (default 1).}

\item{video_connector_num_attention_heads}{Integer. Number of attention heads for video connector.}

\item{video_connector_attention_head_dim}{Integer. Attention head dimension for video.}

\item{video_connector_num_layers}{Integer. Number of transformer layers for video.}

\item{video_connector_num_learnable_registers}{Integer. Number of learnable registers for video.}

\item{audio_connector_num_attention_heads}{Integer. Number of attention heads for audio connector.}

\item{audio_connector_attention_head_dim}{Integer. Attention head dimension for audio.}

\item{audio_connector_num_layers}{Integer. Number of transformer layers for audio.}

\item{audio_connector_num_learnable_registers}{Integer. Number of learnable registers for audio.}

\item{connector_rope_base_seq_len}{Integer. Base sequence length for RoPE.}

\item{rope_theta}{Numeric. RoPE theta parameter.}

\item{rope_double_precision}{Logical. Use double precision for RoPE.}

\item{causal_temporal_positioning}{Logical. Use causal temporal positioning.}

\item{rope_type}{Character. RoPE type ("interleaved" or "split").}
}
\value{
nn_module for text connectors.
}
\description{
Transforms packed text encoder hidden states for video and audio streams.
}
