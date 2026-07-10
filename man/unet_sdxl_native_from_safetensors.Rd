% tinyrox says don't edit this manually, but it can't stop you!
\name{unet_sdxl_native_from_safetensors}
\alias{unet_sdxl_native_from_safetensors}
\title{Build a native SDXL UNet from a diffusers safetensors directory}
\usage{
unet_sdxl_native_from_safetensors(path, verbose = TRUE, ...)
}
\arguments{
\item{path}{Path to the UNet directory or its single-file checkpoint.}

\item{verbose}{Print how many parameters were loaded.}

\item{...}{Overrides for \code{\link{unet_sdxl_native}} constructor args.}
}
\value{
The native SDXL UNet in eval mode.
}
\description{
The safetensors counterpart to
\code{\link{unet_sdxl_native_from_torchscript}}: constructs
\code{\link{unet_sdxl_native}} and loads its weights from
\code{unet/diffusion_pytorch_model.safetensors}. Validated against the
cached \code{stabilityai/stable-diffusion-xl-base-1.0} UNet (all 1680
parameters map with matching shapes). Pass constructor overrides
through \code{...} for a variant checkpoint.
}
