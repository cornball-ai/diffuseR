% tinyrox says don't edit this manually, but it can't stop you!
\name{unet_sdxl_native_from_torchscript}
\alias{unet_sdxl_native_from_torchscript}
\title{Create native SDXL UNet from TorchScript}
\usage{
unet_sdxl_native_from_torchscript(torchscript_path, verbose = TRUE)
}
\arguments{
\item{torchscript_path}{Path to TorchScript SDXL UNet .pt file}

\item{verbose}{Print loading progress}
}
\value{
A native SDXL UNet module with loaded weights
}
\description{
Create native SDXL UNet from TorchScript
}
