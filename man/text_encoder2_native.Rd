% tinyrox says don't edit this manually, but it can't stop you!
\name{text_encoder2_native}
\alias{text_encoder2_native}
\title{Native CLIP Text Encoder 2 (OpenCLIP ViT-bigG for SDXL)}
\usage{
text_encoder2_native(
  vocab_size = 49408,
  context_length = 77,
  embed_dim = 1280,
  num_layers = 32,
  num_heads = 20,
  mlp_dim = 5120
)
}
\arguments{
\item{vocab_size}{Vocabulary size (default 49408)}

\item{context_length}{Maximum sequence length (default 77)}

\item{embed_dim}{Embedding dimension (default 1280)}

\item{num_layers}{Number of transformer layers (default 32)}

\item{num_heads}{Number of attention heads (default 20)}

\item{mlp_dim}{MLP hidden dimension (default 5120)}
}
\value{
An nn_module representing the text encoder
}
\description{
Native R torch implementation of OpenCLIP text encoder used in SDXL.
Returns both hidden states and pooled output.
}
