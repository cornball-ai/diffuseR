% tinyrox says don't edit this manually, but it can't stop you!
\name{gemma4_decoder_layer}
\alias{gemma4_decoder_layer}
\title{Gemma 4 decoder layer}
\usage{
gemma4_decoder_layer(config, layer_idx)
}
\arguments{
\item{config}{Model configuration.}

\item{layer_idx}{Integer. 0-based layer index.}
}
\value{
Module whose forward(hidden_states, attention_mask, cos, sin)
  returns the layer output, scaled by the stored \code{layer_scalar}.
}
\description{
Gemma 4 decoder layer
}
\keyword{internal}
