% tinyrox says don't edit this manually, but it can't stop you!
\name{unet_native_from_torchscript}
\alias{unet_native_from_torchscript}
\title{Create native UNet from TorchScript}
\usage{
unet_native_from_torchscript(torchscript_path, verbose = TRUE)
}
\arguments{
\item{torchscript_path}{Path to TorchScript UNet .pt file}

\item{verbose}{Print loading progress}
}
\value{
A native UNet module with loaded weights
}
\description{
Detects architecture and loads weights from a TorchScript UNet file.
}
