% tinyrox says don't edit this manually, but it can't stop you!
\name{sd_pipeline_from_safetensors}
\alias{sd_pipeline_from_safetensors}
\title{Assemble a native SD pipeline from a diffusers safetensors directory}
\usage{
sd_pipeline_from_safetensors(diffusers_dir, model_name = "sd21",
                             devices = NULL, unet_dtype = NULL, verbose = TRUE)
}
\arguments{
\item{diffusers_dir}{Directory with \code{unet/}, \code{vae/},
\code{text_encoder/} subdirectories.}

\item{model_name}{Currently "sd21" (SDXL pending its second encoder).}

\item{devices}{Named list of component devices (\code{unet},
\code{decoder}, \code{text_encoder}); defaults to all-CPU.}

\item{unet_dtype}{A torch dtype for the UNet (default float16 on CUDA,
float32 on CPU).}

\item{verbose}{Logical.}
}
\value{
A list with \code{unet}, \code{decoder}, \code{text_encoder}.
}
\description{
Builds the native UNet, VAE decoder, and CLIP text encoder from a
diffusers directory using the \code{*_from_safetensors} constructors,
places each on its component device, and returns the \code{$unet /
$decoder / $text_encoder} list the \code{txt2img_*} denoise loop
expects.
}
