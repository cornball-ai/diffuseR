% tinyrox says don't edit this manually, but it can't stop you!
\name{load_text_encoder_weights}
\alias{load_text_encoder_weights}
\title{Load weights from TorchScript text encoder into native encoder}
\usage{
load_text_encoder_weights(native_encoder, torchscript_path, verbose = TRUE)
}
\arguments{
\item{native_encoder}{Native text encoder module}

\item{torchscript_path}{Path to TorchScript encoder .pt file}

\item{verbose}{Print loading progress}
}
\value{
The native encoder with loaded weights (invisibly)
}
\description{
Load weights from TorchScript text encoder into native encoder
}
