% tinyrox says don't edit this manually, but it can't stop you!
\name{gemma3_text_model}
\alias{gemma3_text_model}
\title{Gemma3 Text Model}
\usage{
gemma3_text_model(config)
}
\arguments{
\item{config}{Model configuration list.}
}
\value{
Module whose forward(input_ids, ...) returns
  \code{list(last_hidden_state, hidden_states)}: the final hidden
  state [B, S, hidden_size] and the list of per-layer hidden states.
}
\description{
Full Gemma3 text encoder model.
}
