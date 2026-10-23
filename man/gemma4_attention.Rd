% tinyrox says don't edit this manually, but it can't stop you!
\name{gemma4_attention}
\alias{gemma4_attention}
\title{Gemma 4 text attention}
\usage{
gemma4_attention(config, layer_idx)
}
\arguments{
\item{config}{Model configuration (see \code{\link{load_gemma4_text_encoder}}).}

\item{layer_idx}{Integer. 0-based layer index.}
}
\value{
Module whose forward(hidden_states, attention_mask, cos, sin)
  returns the attention output [B, S, hidden_size].
}
\description{
Gemma 4 text attention
}
\keyword{internal}
