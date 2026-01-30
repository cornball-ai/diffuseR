% tinyrox says don't edit this manually, but it can't stop you!
\name{load_unet_weights}
\alias{load_unet_weights}
\title{Load weights from TorchScript UNet into native UNet}
\usage{
load_unet_weights(native_unet, torchscript_path, verbose = TRUE)
}
\arguments{
\item{native_unet}{Native UNet module}

\item{torchscript_path}{Path to TorchScript UNet .pt file}

\item{verbose}{Print loading progress}
}
\value{
The native UNet with loaded weights (invisibly)
}
\description{
Load weights from TorchScript UNet into native UNet
}
