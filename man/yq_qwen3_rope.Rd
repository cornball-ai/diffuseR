% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_qwen3_rope}
\alias{yq_qwen3_rope}
\title{Qwen3 rotary tables (host-side, Llama convention)}
\usage{
yq_qwen3_rope(seq_len, head_dim, theta = 1e+06, device = "cpu")
}
\arguments{
\item{seq_len}{Integer. Sequence length.}

\item{head_dim}{Integer. Per-head dim (128).}

\item{theta}{Numeric. RoPE base (Qwen3: 1e6).}

\item{device}{Character.}
}
\value{
List \code{list(cos, sin)}, each \code{[S, head_dim/2]}, f32.
}
\description{
\code{inv_freq = 1 / theta^(arange(0, D-2, 2) / D)}; cos/sin of the
position outer product. Returned as \code{[S, D/2]} AnvlArrays for the
split-half kernel (\code{\link[yunque]{yq_rope_split}}).
}
