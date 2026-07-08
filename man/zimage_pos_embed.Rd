% tinyrox says don't edit this manually, but it can't stop you!
\name{zimage_pos_embed}
\alias{zimage_pos_embed}
\title{Compute Z-Image rotary frequencies from position ids}
\usage{
zimage_pos_embed(ids, axes_dim = c(32L, 48L, 48L), theta = 256)
}
\arguments{
\item{ids}{Tensor of shape [S, 3] from \code{zimage_cap_pos_ids} /
\code{zimage_img_pos_ids}.}

\item{axes_dim}{Integer vector of per-axis rotary dims; must sum to
the attention head dim. Z-Image uses c(32, 48, 48).}

\item{theta}{Numeric. RoPE base frequency. Z-Image uses 256.}
}
\value{
List of two tensors (cos, sin), each [S, sum(axes_dim)],
  float32, on the device of \code{ids}.
}
\description{
Per-axis 1D rotary frequencies in the interleaved-real convention.
Frequencies and angles are built in float64, then the angles are cast
to float32 before cos/sin — matching the reference torch.polar call on
a .float() tensor. Output format matches \code{flux_pos_embed} so
\code{flux_apply_rotary_emb} applies unchanged.
}
