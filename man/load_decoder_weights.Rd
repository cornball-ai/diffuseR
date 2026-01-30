% tinyrox says don't edit this manually, but it can't stop you!
\name{load_decoder_weights}
\alias{load_decoder_weights}
\title{Load weights from TorchScript decoder into native decoder}
\usage{
load_decoder_weights(native_decoder, torchscript_path, verbose = TRUE)
}
\arguments{
\item{native_decoder}{Native VAE decoder module}

\item{torchscript_path}{Path to TorchScript decoder .pt file}

\item{verbose}{Print loading progress}
}
\value{
The native decoder with loaded weights (invisibly)
}
\description{
Load weights from TorchScript decoder into native decoder
}
