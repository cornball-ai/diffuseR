% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_ltx_video_rope}
\alias{yq_ltx_video_rope}
\title{LTX-2.3 video RoPE cos/sin tables (host-side)}
\usage{
yq_ltx_video_rope(num_frames, height, width, num_heads, head_dim, patch = 1L,
                  patch_t = 1L, base_num_frames = 20L, base_height = 2048L,
                  base_width = 2048L, scale_factors = c(8L, 32L, 32L),
                  theta = 10000, causal_offset = 1L, fps = 24, batch = 1L,
                  device = "cpu")
}
\arguments{
\item{scale_factors}{Integer vector (time, height, width) VAE scales.}

\item{theta}{Numeric. RoPE theta.}

\item{causal_offset}{Integer. Causal-VAE temporal offset.}

\item{fps}{Numeric. Frames per second.}

\item{batch}{Integer. Batch size.}

\item{device}{Character. Target device.}

\item{num_frames,height,width}{Integers. Latent token grid.}

\item{num_heads,head_dim}{Video attention shape (\code{dim = num_heads *
head_dim}).}

\item{patch,patch_t}{Integers. Spatial/temporal patch sizes.}

\item{base_num_frames,base_height,base_width}{Integers. Normalization grid.}
}
\value{
List \code{list(cos, sin)}, each \code{[batch, num_heads, N,
  head_dim/2]}, f32.
}
\description{
Base-R port of \code{ltx23_rotary_pos_embed$prepare_video_coords} +
\code{$forward} for the "split" layout: patch-boundary coordinates
(frames scaled to seconds via fps with a causal temporal shift, height
and width in pixel space), midpoints normalized against the base grid,
outer product with the rope frequencies, reshaped per attention head.
}
