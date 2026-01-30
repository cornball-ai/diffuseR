% tinyrox says don't edit this manually, but it can't stop you!
\name{load_unet_sdxl_weights}
\alias{load_unet_sdxl_weights}
\title{Load weights from TorchScript SDXL UNet into native SDXL UNet}
\usage{
load_unet_sdxl_weights(native_unet, torchscript_path, verbose = TRUE)
}
\arguments{
\item{native_unet}{Native SDXL UNet module}

\item{torchscript_path}{Path to TorchScript SDXL UNet .pt file}

\item{verbose}{Print loading progress}
}
\value{
The native UNet with loaded weights (invisibly)
}
\description{
Load weights from TorchScript SDXL UNet into native SDXL UNet
}
