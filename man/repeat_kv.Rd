% tinyrox says don't edit this manually, but it can't stop you!
\name{repeat_kv}
\alias{repeat_kv}
\title{Repeat KV heads for GQA (Grouped Query Attention)}
\usage{
repeat_kv(hidden_states, n_rep)
}
\arguments{
\item{hidden_states}{Tensor of shape [batch, num_kv_heads, seq_len, head_dim]}

\item{n_rep}{Number of repetitions per KV head}
}
\value{
Tensor of shape [batch, num_kv_heads * n_rep, seq_len, head_dim]
}
\description{
Interleaves KV heads to match query heads: [k0,k0,k1,k1,...] not [k0,k1,...,k0,k1,...]
}
\keyword{internal}
