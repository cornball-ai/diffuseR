\name{load_model_component}
\alias{load_model_component}
\title{Load a specific component of a diffusion model}
\description{Loads a TorchScript model component (UNet, decoder, or text encoder) from the local model directory, downloading it first if necessary.}
\usage{
load_model_component(component, model_name, device, unet_dtype_str, download, use_native)
}
\arguments{
  \item{component}{Character string, the component to load: "unet", "decoder", or "text_encoder".}
  \item{model_name}{Character string, the name of the model to use.}
  \item{device}{Character string, the torch device to load the model onto ("cpu" or "cuda").}
  \item{unet_dtype_str}{Optional; the data type for the UNet model. If `NULL`, defaults to `float32` for CPU and `float16` for CUDA.}
  \item{download}{Logical; if `TRUE` (default), downloads the model if it doesn't exist locally.}
  \item{use_native}{Logical; if `TRUE`, uses native R torch modules instead of TorchScript. Supported for unet, decoder, text_encoder, and text_encoder2. Native modules have better GPU compatibility (especially on Blackwell/RTX 50xx).}
}
\value{
A torch model object.
}
\examples{
\dontrun{
unet <- load_model_component("unet", "sd21", "cpu")
}
}
