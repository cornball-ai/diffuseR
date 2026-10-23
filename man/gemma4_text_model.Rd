% tinyrox says don't edit this manually, but it can't stop you!
\name{gemma4_text_model}
\alias{gemma4_text_model}
\title{Gemma 4 text model}
\usage{
gemma4_text_model(config)
}
\arguments{
\item{config}{Model configuration list: the checkpoint's
\code{text_config} fields (see \code{\link{load_gemma4_text_encoder}}).}
}
\value{
Module whose forward(input_ids, attention_mask,
  output_hidden_states) returns \code{list(last_hidden_state,
  hidden_states)}. \code{input_ids} are 0-based; \code{hidden_states}
  follows transformers: the state entering each layer, then the
  final-normed output (\code{num_hidden_layers + 1} entries).
}
\description{
The text stack of Gemma 4 Unified, as the LTX-2.5 text encoder.
}
