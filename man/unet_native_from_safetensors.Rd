% tinyrox says don't edit this manually, but it can't stop you!
\name{unet_native_from_safetensors}
\alias{unet_native_from_safetensors}
\title{Build a native SD21 UNet from a diffusers safetensors directory}
\usage{
unet_native_from_safetensors(path, verbose = TRUE, ...)
}
\arguments{
\item{path}{Path to the UNet directory or its single-file checkpoint.}

\item{verbose}{Print how many parameters were loaded.}

\item{...}{Overrides for \code{\link{unet_native}} constructor args.}
}
\value{
The native SD21 UNet in eval mode.
}
\description{
The safetensors counterpart to
\code{\link{unet_native_from_torchscript}}: constructs
\code{\link{unet_native}} and loads its weights from
\code{unet/diffusion_pytorch_model.safetensors} (no TorchScript, so it
works on Blackwell). The default construction matches the canonical
Stable Diffusion 2.1 UNet; pass constructor overrides through
\code{...} for a variant checkpoint (the loader fails loudly on any
shape mismatch, so a wrong architecture surfaces immediately rather
than loading silently wrong weights).
}
