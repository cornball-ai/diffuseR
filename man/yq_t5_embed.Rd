% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_t5_embed}
\alias{yq_t5_embed}
\title{T5 token embedding lookup (host-side gather)}
\usage{
yq_t5_embed(embed, ids, device = "cpu")
}
\arguments{
\item{embed}{R matrix \code{[vocab, d_model]} (from
\code{\link{yq_t5_load_weights}}).}

\item{ids}{Integer matrix \code{[B, S]} of 0-based token ids.}

\item{device}{Character.}
}
\value{
AnvlArray \code{[B, S, d_model]}, f32.
}
\description{
Gathers rows of the shared embedding table for the given token ids.
The table stays an R matrix (never resident on device); only the
gathered \code{[B, S, d_model]} result crosses to anvl.
}
