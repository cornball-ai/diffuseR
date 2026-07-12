% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_ltx_audio_causal_conv2d}
\alias{yq_ltx_audio_causal_conv2d}
\title{Causal 2D convolution for audio spectrograms (anvl)}
\usage{
yq_ltx_audio_causal_conv2d(x, weight, bias, causality_axis = "height")
}
\arguments{
\item{x}{AnvlArray \code{[B, C_in, T, M]}.}

\item{weight}{AnvlArray \code{[C_out, C_in, k_h, k_w]} (raw torch layout).}

\item{bias}{AnvlArray \code{[C_out]}.}

\item{causality_axis}{Character. "height" (default), "width", or "none".}
}
\value{
AnvlArray \code{[B, C_out, T, M]}.
}
\description{
Zero-pads asymmetrically before an unpadded stride-1 Conv2d, matching
\code{diffuseR::ltx23_audio_causal_conv2d}. For \code{causality_axis =
"height"} the time axis (dim 3) is padded by \code{k_h-1} on the
leading side only; the mel axis (dim 4) gets symmetric padding.
}
