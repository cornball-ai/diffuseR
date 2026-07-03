% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_rotary_pos_embed}
\alias{ltx23_rotary_pos_embed}
\title{LTX-2.3 audio/video rotary position embedder}
\usage{
ltx23_rotary_pos_embed(dim, patch_size = 1L, patch_size_t = 1L,
                       base_num_frames = 20L, base_height = 2048L,
                       base_width = 2048L, sampling_rate = 16000L,
                       hop_length = 160L, scale_factors = c(8L, 32L, 32L),
                       theta = 10000, causal_offset = 1L, modality = "video",
                       double_precision = TRUE, rope_type = "split",
                       num_attention_heads = 32L)
}
\arguments{
\item{dim}{Integer. Rotary dimension (attention head dim x heads for
split type at model level; see reference).}

\item{scale_factors}{Integer vector. VAE (time, height, width) scale
factors.}

\item{theta}{Numeric. RoPE theta.}

\item{causal_offset}{Integer. Temporal offset for the causal VAE
(first frame has stride 1).}

\item{modality}{"video" or "audio".}

\item{double_precision}{Logical. Compute base frequencies in float64.}

\item{rope_type}{"split" (LTX 2.3) or "interleaved".}

\item{num_attention_heads}{Integer. Needed for the split layout.}

\item{patch_size,patch_size_t}{Integers. Spatial/temporal patch sizes.}

\item{base_num_frames,base_height,base_width}{Integers. Base grid the
coordinates are normalized against.}

\item{sampling_rate,hop_length}{Integers. Audio spectrogram params.}
}
\description{
Computes RoPE cos/sin frequency tensors from spatiotemporal patch
coordinates. Video coordinates are 3D (frames scaled to seconds via
fps, height, width in pixel space); audio coordinates are 1D
(seconds). Coordinates are patch boundaries [start, end); the midpoint
is used as the position.
}
