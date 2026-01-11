\name{detect_text_encoder_architecture}
\alias{detect_text_encoder_architecture}
\title{Detect text encoder architecture from TorchScript file}
\description{Detect text encoder architecture from TorchScript file}
\usage{
detect_text_encoder_architecture(torchscript_path)
}
\arguments{
  \item{torchscript_path}{Path to TorchScript encoder .pt file}
}
\value{
List with vocab_size, context_length, embed_dim, num_layers, num_heads, mlp_dim
}
\keyword{internal}
