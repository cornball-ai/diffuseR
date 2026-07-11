% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_qwen3_embed}
\alias{yq_qwen3_embed}
\title{Qwen3 token embedding lookup (host-side gather)}
\usage{
yq_qwen3_embed(embed, ids, device = "cpu")
}
\arguments{
\item{embed}{R matrix \code{[vocab, hidden]} (from
\code{\link{yq_qwen3_load_weights}}).}

\item{ids}{Integer matrix \code{[B, S]} of 0-based token ids.}

\item{device}{Character.}
}
\value{
AnvlArray \code{[B, S, hidden]}, f32.
}
\description{
Gathers rows of the embedding table for the given token ids. The
table stays an R matrix (never a resident device tensor); the result
is the only thing that crosses to anvl.
}
