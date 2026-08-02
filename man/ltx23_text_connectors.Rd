% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_text_connectors}
\alias{ltx23_text_connectors}
\title{LTX-2.3 text connectors}
\usage{
ltx23_text_connectors(caption_channels = 3840L, text_proj_in_factor = 49L,
                      video_connector_num_attention_heads = 32L,
                      video_connector_attention_head_dim = 128L,
                      video_connector_num_layers = 8L,
                      video_connector_num_learnable_registers = 128L,
                      video_gated_attn = TRUE,
                      audio_connector_num_attention_heads = 32L,
                      audio_connector_attention_head_dim = 64L,
                      audio_connector_num_layers = 8L,
                      audio_connector_num_learnable_registers = 128L,
                      audio_gated_attn = TRUE,
                      connector_rope_base_seq_len = 4096L, rope_theta = 10000,
                      rope_double_precision = TRUE, rope_type = "split",
                      video_hidden_dim = 4096L, audio_hidden_dim = 2048L,
                      proj_bias = TRUE)
}
\arguments{
\item{caption_channels}{Integer. Text encoder hidden size (3840 for
Gemma3-12B).}

\item{text_proj_in_factor}{Integer. Number of stacked hidden states
(num_layers + 1 = 49 for Gemma3-12B).}

\item{video_connector_num_learnable_registers}{Integer or NULL.}

\item{video_gated_attn}{Logical.}

\item{audio_connector_num_learnable_registers}{Integer or NULL.}

\item{audio_gated_attn}{Logical.}

\item{proj_bias}{Logical. Projection bias (TRUE for LTX-2.3).}

\item{video_connector_num_attention_heads,video_connector_attention_head_dim,video_connector_num_layers}{Video connector shape (LTX-2.3: 32 x 128, 8 layers).}

\item{audio_connector_num_attention_heads,audio_connector_attention_head_dim,audio_connector_num_layers}{Audio connector shape (LTX-2.3: 32 x 64, 8 layers).}

\item{connector_rope_base_seq_len,rope_theta,rope_double_precision,rope_type}{RoPE config.}

\item{video_hidden_dim,audio_hidden_dim}{Integers. Projection targets
(DiT inner dims: 4096 / 2048).}
}
\value{
Module whose forward(text_encoder_hidden_states,
  attention_mask) returns \code{list(video_text_embedding,
  audio_text_embedding, attention_mask)}: the caption states adapted
  for the video and audio cross-attention streams, plus the binary
  mask to use with them.
}
\description{
Takes raw stacked per-layer text encoder hidden states and produces
the video and audio text embeddings for the DiT: per-token RMS norm,
per-modality sqrt(dim ratio) rescaling and projection, then a
per-modality 1D connector transformer.
}
