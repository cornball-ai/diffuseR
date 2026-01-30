% tinyrox says don't edit this manually, but it can't stop you!
\name{gemma3_attention}
\alias{gemma3_attention}
\title{Gemma3 Attention}
\usage{
gemma3_attention(config, layer_idx = 0L)
}
\arguments{
\item{config}{Model configuration.}

\item{layer_idx}{Integer. Layer index for layer-specific settings.}
}
\description{
Multi-head attention with Grouped Query Attention (GQA) and optional
sliding window attention.
}
\keyword{internal}
