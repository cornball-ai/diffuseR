% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_gemma3_rope}
\alias{yq_gemma3_rope}
\title{Gemma3 rotary tables (host-side, split-half convention)}
\usage{
yq_gemma3_rope(seq_len, head_dim, theta = 1e+06, scaling_factor = 1,
               device = "cpu")
}
\arguments{
\item{seq_len}{Integer. Sequence length.}

\item{head_dim}{Integer. Per-head dim.}

\item{theta}{Numeric. RoPE base (global 1e6, local 1e4).}

\item{scaling_factor}{Numeric. Linear RoPE scaling (global 8, local 1);
\code{inv_freq} is divided by it (HuggingFace convention).}

\item{device}{Character.}
}
\value{
List \code{list(cos, sin)}, each \code{[S, head_dim/2]}, f32.
}
\description{
\code{inv_freq = (1 / theta^(arange(0, D-2, 2) / D)) / scaling_factor};
cos/sin of the position outer product. Gemma3 uses TWO of these: a
global table (theta 1e6, inv_freq divided by the rope_scaling factor)
for the every-6th "global" layer, and a local table (theta 1e4, no
scaling) for the sliding-window layers. Returned as \code{[S, D/2]}
AnvlArrays for the split-half kernel (\code{\link[yunque]{yq_rope_split}}).
}
