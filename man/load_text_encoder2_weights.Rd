\name{load_text_encoder2_weights}
\alias{load_text_encoder2_weights}
\title{Load weights from TorchScript text encoder 2 into native encoder}
\description{Load weights from TorchScript text encoder 2 into native encoder}
\usage{
load_text_encoder2_weights(native_encoder, torchscript_path, verbose)
}
\arguments{
  \item{native_encoder}{Native text encoder 2 module}
  \item{torchscript_path}{Path to TorchScript encoder .pt file}
  \item{verbose}{Print loading progress}
}
\value{
The native encoder with loaded weights (invisibly)
}
