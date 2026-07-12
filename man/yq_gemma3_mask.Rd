% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_gemma3_mask}
\alias{yq_gemma3_mask}
\title{Gemma3 additive attention mask (host-side)}
\usage{
yq_gemma3_mask(attention_mask, seq_len, batch = 1L, window = NULL,
               device = "cpu", neg = -1e+09)
}
\arguments{
\item{attention_mask}{Integer/numeric matrix \code{[B, S]} (1 real,
0 pad), or NULL for causal-only.}

\item{seq_len}{Integer. Sequence length.}

\item{batch}{Integer. Batch size.}

\item{window}{Integer sliding-window size, or NULL for full causal.}

\item{device}{Character.}

\item{neg}{Numeric. Per-masked-condition additive bias.}
}
\value{
AnvlArray \code{[B, 1, S, S]}, f32.
}
\description{
Causal upper-triangular mask plus per-token padding, as an additive
bias \code{[B, 1, S, S]} (0 where attended, \code{neg} where masked),
mirroring the reference's \code{(causal + padding) * -1e9}. When
\code{window} is set, keys more than \code{window} positions in the
past are also masked (the local sliding variant). The reference
\code{gemma3_text_model} passes the full (\code{window = NULL}) mask to
every layer — its \code{create_sliding_window_mask} is never called —
so the encoder uses the full mask; the sliding path is exposed here for
completeness. Broadcasts against scores inside \code{\link[yunque]{yq_sdpa}}.
}
