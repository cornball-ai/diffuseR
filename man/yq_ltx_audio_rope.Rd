% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_ltx_audio_rope}
\alias{yq_ltx_audio_rope}
\title{LTX-2.3 audio RoPE cos/sin tables (host-side)}
\usage{
yq_ltx_audio_rope(num_frames, num_heads, head_dim, patch_t = 1L,
                  base_num_frames = 20L, scale_factor = 4L,
                  sampling_rate = 16000L, hop_length = 160L, theta = 10000,
                  causal_offset = 1L, batch = 1L, device = "cpu")
}
\arguments{
\item{num_frames}{Integer. Audio latent token count.}

\item{patch_t}{Integer. Temporal patch size.}

\item{base_num_frames}{Integer. Normalization grid.}

\item{scale_factor}{Integer. Audio VAE scale factor.}

\item{theta}{Numeric. RoPE theta.}

\item{causal_offset}{Integer. Causal-VAE temporal offset.}

\item{batch}{Integer. Batch size.}

\item{device}{Character. Target device.}

\item{num_heads,head_dim}{Audio attention shape.}

\item{sampling_rate,hop_length}{Integers. Mel spectrogram params.}
}
\value{
List \code{list(cos, sin)}, each \code{[batch, num_heads, N,
  head_dim/2]}, f32.
}
\description{
Base-R port of \code{ltx23_rotary_pos_embed$prepare_audio_coords} +
\code{$forward} for the audio (1-D, seconds) modality.
}
