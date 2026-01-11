\name{unet_sdxl_native_from_torchscript}
\alias{unet_sdxl_native_from_torchscript}
\title{Create native SDXL UNet from TorchScript}
\description{Create native SDXL UNet from TorchScript}
\usage{
unet_sdxl_native_from_torchscript(torchscript_path, verbose)
}
\arguments{
  \item{torchscript_path}{Path to TorchScript SDXL UNet .pt file}
  \item{verbose}{Print loading progress}
}
\value{
A native SDXL UNet module with loaded weights
}
