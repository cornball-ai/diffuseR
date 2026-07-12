% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_sdxl_clip_mask}
\alias{yq_sdxl_clip_mask}
\title{SDXL CLIP causal additive attention mask (host-side)}
\usage{
yq_sdxl_clip_mask(seq_len, batch = 1L, device = "cpu", neg = -3.4e+38)
}
\arguments{
\item{seq_len}{Integer. Sequence length.}

\item{batch}{Integer. Batch size.}

\item{device}{Character. Target device.}

\item{neg}{Numeric. Masked-position bias.}
}
\value{
AnvlArray \code{[B, 1, S, S]}, f32.
}
\description{
Upper-triangular causal mask as an additive bias \code{[B, 1, S, S]}
(0 where attended, a large negative where key j > query i). SDXL feeds
no padding mask (CLIP causal attention makes trailing pads irrelevant),
so this is causal-only. Broadcasts against scores inside
\code{\link[yunque]{yq_sdpa}}.
}
