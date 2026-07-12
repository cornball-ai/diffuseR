% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_ltx_text_mask}
\alias{yq_ltx_text_mask}
\title{LTX-2.3 text cross-attention mask (host-side)}
\usage{
yq_ltx_text_mask(mask, device = "cpu", neg = -10000)
}
\arguments{
\item{mask}{Numeric/integer matrix \code{[B, S]}, or NULL (no mask).}

\item{device}{Character. Target device.}

\item{neg}{Numeric. Masked-position bias (reference uses -10000).}
}
\value{
AnvlArray \code{[B, 1, 1, S]}, f32, or NULL.
}
\description{
Converts a multiplicative padding mask \code{[B, S]} (1 real, 0 pad)
into the additive attention bias the score matrix wants, matching the
reference \code{(1 - mask) * -10000} expanded to \code{[B, 1, 1, S]} so
it broadcasts against scores \code{[B, H, Sq, S]} inside
\code{\link[yunque]{yq_sdpa}}.
}
