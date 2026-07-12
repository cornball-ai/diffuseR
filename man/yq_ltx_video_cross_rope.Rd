% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_ltx_video_cross_rope}
\alias{yq_ltx_video_cross_rope}
\title{LTX-2.3 video->audio cross-attention RoPE tables (host-side)}
\usage{
yq_ltx_video_cross_rope(num_frames, height, width, num_heads, cross_dim,
                        patch = 1L, patch_t = 1L, base_num_frames = 20L,
                        scale_factor = 8L, theta = 10000, causal_offset = 1L,
                        fps = 24, batch = 1L, device = "cpu")
}
\arguments{
\item{num_heads}{Integer. Video attention heads.}

\item{cross_dim}{Integer. Cross-attention dimension (audio inner dim).}

\item{base_num_frames}{Integer. Temporal normalization grid.}

\item{scale_factor}{Integer. VAE temporal scale (\code{scale_factors[1]}).}

\item{theta}{Numeric. RoPE theta.}

\item{causal_offset}{Integer. Causal-VAE temporal offset.}

\item{fps}{Numeric. Frames per second.}

\item{batch}{Integer. Batch size.}

\item{device}{Character. Target device.}

\item{num_frames,height,width}{Integers. Video latent token grid (only
\code{num_frames} drives the temporal coordinate; height/width set N).}

\item{patch,patch_t}{Integers. Spatial/temporal patch sizes.}
}
\value{
List \code{list(cos, sin)}, each \code{[batch, num_heads, N,
  cross_dim/(2*num_heads)]}, f32.
}
\description{
The audio<->video cross-attention rotates only the temporal axis. This
is the video side: a 1-axis rope over the video temporal midpoints
(normalized by \code{base_num_frames}) at the cross-attention dimension
and the video head count (\code{ltx23_transformer$cross_attn_rope}). The
audio-side cross rope is identical to \code{\link{yq_ltx_audio_rope}}
(same dim, heads, and 1-axis audio coords).
}
