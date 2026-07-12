% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_t5_rel_pos_bias}
\alias{yq_t5_rel_pos_bias}
\title{T5 relative-position bias (host-side, bucketed)}
\usage{
yq_t5_rel_pos_bias(rel_bias_weight, seq_len, num_buckets = 32L,
                   max_distance = 128L, device = "cpu")
}
\arguments{
\item{rel_bias_weight}{R matrix \code{[num_buckets, num_heads]} (the
\code{relative_attention_bias} embedding, from
\code{\link{yq_t5_load_weights}}).}

\item{seq_len}{Integer. Sequence length.}

\item{num_buckets}{Integer. Total relative-position buckets (32).}

\item{max_distance}{Integer. Max distance before log-spacing saturates
(128).}

\item{device}{Character.}
}
\value{
AnvlArray \code{[1, num_heads, seq_len, seq_len]}, f32.
}
\description{
Reproduces \code{T5Attention.compute_bias}: bucket the bidirectional
relative positions \eqn{j - i} (half the buckets split by sign, half
of each side exact small offsets, the rest log-spaced up to
\code{max_distance}), gather block 1's learned bias embedding, and lay
it out as the additive \code{[1, H, S, S]} tensor added to every
layer's attention logits. Parameter-free integer indexing, so it is
computed in base R and passed into the jitted encoder as an input
(same boundary logic as the token embedding gather).
}
