\name{load_pipeline}
\alias{load_pipeline}
\title{Load a diffusion model pipeline}
\description{This function loads a diffusion model pipeline consisting of a UNet, VAE decoder, and text encoder. It initializes the models and sets up the environment for inference.}
\usage{
load_pipeline(model_name, m2d, i2i, unet_dtype_str, use_native_decoder, use_native_text_encoder, use_native_unet, ...)
}
\arguments{
  \item{model_name}{The name of the model to load.}
  \item{m2d}{A list containing model-to-device mappings and configurations.}
  \item{i2i}{Logical indicating whether to load the encoder for img2img().}
  \item{unet_dtype_str}{A string representing the data type for the UNet model (e.g., "float32", "float16").}
  \item{use_native_decoder}{Logical; if TRUE, uses native R torch decoder instead of TorchScript. Native decoder has better GPU compatibility (especially Blackwell).}
  \item{use_native_text_encoder}{Logical; if TRUE, uses native R torch text encoder instead of TorchScript. Native text encoder has better GPU compatibility (especially Blackwell).}
  \item{use_native_unet}{Logical; if TRUE, uses native R torch UNet instead of TorchScript. Native UNet has better GPU compatibility (especially Blackwell).}
  \item{...}{Additional arguments passed to the model loading functions.}
}
\value{
An environment containing the loaded models and configuration.
}
\examples{
\dontrun{
pipeline <- load_pipeline("my_model", device = "cuda")
}

}
