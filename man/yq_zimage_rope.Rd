% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_zimage_rope}
\alias{yq_zimage_rope}
\title{Z-Image RoPE tables for a txt2img forward (host-side)}
\usage{
yq_zimage_rope(h_tokens, w_tokens, cap_len, f_tokens = 1L,
               axes_dim = c(32L, 48L, 48L), theta = 256, seq_multi = 32L,
               device = "cpu")
}
\arguments{
\item{h_tokens}{Integer. Token grid height (latent height / patch).}

\item{w_tokens}{Integer. Token grid width (latent width / patch).}

\item{cap_len}{Integer. Caption length before padding.}

\item{f_tokens}{Integer. Token grid frames (1 for txt2img).}

\item{axes_dim}{Integer vector of per-axis rotary dims (must sum to the
attention head dim; full model: \code{c(32, 48, 48)}).}

\item{theta}{Numeric. Base frequency (256).}

\item{seq_multi}{Integer. Sub-sequence padding multiple (32).}

\item{device}{Character. Target device.}
}
\value{
List \code{list(cos_img, sin_img, cos_cap, sin_cap)}, each an
  \code{AnvlArray} \code{[padded_len, sum(axes_dim)]}, f32.
}
\description{
Base-R port of the Z-Image position scheme
(\code{diffuseR::zimage_cap_pos_ids} / \code{zimage_img_pos_ids} +
\code{zimage_pos_embed}). Caption tokens ramp \code{1..cap_padded} on
axis 1; image tokens carry the frame index (offset just past the
caption) on axis 1 and the token row/column on axes 2/3. Each
sub-sequence is padded to a multiple of 32. Returns separate image and
caption tables (the main trunk concatenates them, image first).
}
