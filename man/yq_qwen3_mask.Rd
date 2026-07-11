% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_qwen3_mask}
\alias{yq_qwen3_mask}
\title{Qwen3 additive attention mask (host-side)}
\usage{
yq_qwen3_mask(attention_mask, seq_len, batch = 1L, device = "cpu",
              neg = -3.4e+38)
}
\arguments{
\item{attention_mask}{Integer/numeric matrix \code{[B, S]} (1 real,
0 pad), or NULL for causal-only.}

\item{seq_len}{Integer. Sequence length (rows of attention_mask).}

\item{batch}{Integer. Batch size.}

\item{device}{Character.}

\item{neg}{Numeric. Masked-position bias.}
}
\value{
AnvlArray \code{[B, 1, S, S]}, f32.
}
\description{
Causal upper-triangular mask plus per-token padding, as an additive
bias \code{[B, 1, S, S]} (0 where attended, a large negative where
masked). Broadcasts against scores inside \code{\link[yunque]{yq_sdpa}}.
}
