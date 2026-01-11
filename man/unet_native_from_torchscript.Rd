\name{unet_native_from_torchscript}
\alias{unet_native_from_torchscript}
\title{Create native UNet from TorchScript}
\description{Detects architecture and loads weights from a TorchScript UNet file.}
\usage{
unet_native_from_torchscript(torchscript_path, verbose)
}
\arguments{
  \item{torchscript_path}{Path to TorchScript UNet .pt file}
  \item{verbose}{Print loading progress}
}
\value{
A native UNet module with loaded weights
}
