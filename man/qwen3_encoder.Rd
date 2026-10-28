% tinyrox says don't edit this manually, but it can't stop you!
\name{qwen3_encoder}
\alias{qwen3_encoder}
\title{Qwen3 encoder stack}
\usage{
qwen3_encoder(
  vocab_size = 151936L,
  hidden_size = 2560L,
  intermediate_size = 9728L,
  num_hidden_layers = 36L,
  num_attention_heads = 32L,
  num_key_value_heads = 8L,
  head_dim = 128L,
  rope_theta = 1e+06,
  rms_norm_eps = 1e-06
)
}
\arguments{
\item{rope_theta}{Numeric.}

\item{rms_norm_eps}{Numeric.}

\item{vocab_size,hidden_size,intermediate_size,num_hidden_layers}{Integers.}

\item{num_attention_heads,num_key_value_heads,head_dim}{Integers.}
}
\value{
Module whose forward(input_ids, attention_mask = NULL,
  out_layers, position_ids = NULL) returns a list of hidden-state
  tensors [B, S, hidden], one per requested layer (a value of k means
  the state after k layers, matching HF \code{output.hidden_states[k]},
  which carries the final norm when k is the last layer).
  Runs only to \code{max(out_layers)}. \code{input_ids} are 1-based;
  \code{position_ids} (0-based integers, length S, shared by the batch)
  override the default 0..S-1 rotary positions.
}
\description{
Defaults are the Qwen3-4B configuration used by FLUX.2 klein. The
module tree mirrors the checkpoint keys (\code{model.embed_tokens},
\code{model.layers.*}, \code{model.norm}); the tied LM head carries
no weights of its own and is not implemented.
}
