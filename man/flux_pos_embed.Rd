% tinyrox says don't edit this manually, but it can't stop you!
\name{flux_pos_embed}
\alias{flux_pos_embed}
\title{Compute FLUX rotary frequencies from position ids}
\usage{
flux_pos_embed(ids, axes_dim = c(16L, 56L, 56L), theta = 10000)
}
\arguments{
\item{ids}{Tensor of shape [S, 3]: concatenated text ids (all zero)
and image ids from \code{flux_prepare_latent_image_ids}.}

\item{axes_dim}{Integer vector of per-axis rotary dims; must sum to
the attention head dim. FLUX uses c(16, 56, 56).}

\item{theta}{Numeric. RoPE base frequency.}
}
\value{
List of two tensors (cos, sin), each [S, sum(axes_dim)],
  float32, on the device of \code{ids}.
}
\description{
Per-axis 1D rotary frequencies (interleaved-real convention), computed
in float64 on CPU and concatenated over the axes. Reference:
FluxPosEmbed with get_1d_rotary_pos_embed(repeat_interleave_real=TRUE,
use_real=TRUE, freqs_dtype=float64).
}
