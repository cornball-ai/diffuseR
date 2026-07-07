% tinyrox says don't edit this manually, but it can't stop you!
\name{t5_encoder}
\alias{t5_encoder}
\title{T5 encoder stack}
\usage{
t5_encoder(vocab_size = 32128L, d_model = 4096L, d_kv = 64L, num_heads = 64L,
           d_ff = 10240L, num_layers = 24L,
           relative_attention_num_buckets = 32L,
           relative_attention_max_distance = 128L, layer_norm_epsilon = 1e-06)
}
\arguments{
\item{layer_norm_epsilon}{Numeric.}

\item{vocab_size,d_model,d_kv,num_heads,d_ff,num_layers}{Integers.}

\item{relative_attention_num_buckets,relative_attention_max_distance}{Integers. Relative position bias shape.}
}
\value{
Module whose forward(input_ids) (1-based ids [B, S]) returns
  the last hidden state [B, S, d_model].
}
\description{
Defaults are the T5-v1.1-XXL configuration used by FLUX.
}
