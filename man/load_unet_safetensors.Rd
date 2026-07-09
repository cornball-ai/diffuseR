% tinyrox says don't edit this manually, but it can't stop you!
\name{load_unet_safetensors}
\alias{load_unet_safetensors}
\title{Load HF safetensors weights into the native SD21 UNet}
\usage{
load_unet_safetensors(native_unet, path, verbose = TRUE)
}
\arguments{
\item{native_unet}{A \code{\link{unet_native}} module.}

\item{path}{Path to the UNet directory (containing
\code{diffusion_pytorch_model.safetensors} or its shard index) or
directly to the single-file checkpoint.}

\item{verbose}{Print how many parameters were loaded.}
}
\value{
The native UNet with weights loaded (invisibly).
}
\description{
Load HF safetensors weights into the native SD21 UNet
}
