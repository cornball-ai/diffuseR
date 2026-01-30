% tinyrox says don't edit this manually, but it can't stop you!
\name{apply_rotary_pos_emb}
\alias{apply_rotary_pos_emb}
\title{Apply rotary position embeddings}
\usage{
apply_rotary_pos_emb(q, k, cos, sin)
}
\arguments{
\item{q}{Query tensor [batch, heads, seq, head_dim]}

\item{k}{Key tensor [batch, heads, seq, head_dim]}

\item{cos}{Cosine embeddings [batch, seq, head_dim]}

\item{sin}{Sine embeddings [batch, seq, head_dim]}
}
\description{
Apply rotary position embeddings
}
\keyword{internal}
