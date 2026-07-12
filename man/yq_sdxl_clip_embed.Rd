% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_sdxl_clip_embed}
\alias{yq_sdxl_clip_embed}
\title{SDXL CLIP token + positional embedding lookup (host-side gather)}
\usage{
yq_sdxl_clip_embed(token_embedding, position_embedding, ids, device = "cpu")
}
\arguments{
\item{token_embedding}{R matrix \code{[vocab, hidden]} (from
\code{\link{yq_sdxl_clip_load_weights}}).}

\item{position_embedding}{R matrix \code{[max_pos, hidden]}.}

\item{ids}{Integer matrix \code{[B, S]} of 0-based token ids.}

\item{device}{Character. Target device.}
}
\value{
AnvlArray \code{[B, S, hidden]}, f32.
}
\description{
Gathers the token-embedding rows for the given ids and adds the learned
positional-embedding rows for positions \code{1..S}. Both tables stay R
matrices (never resident device tensors); only the summed
\code{[B, S, hidden]} result crosses to anvl. Mirrors the native
encoders' \code{token_embedding(ids + 1) + position_embedding[1:S]}.
}
