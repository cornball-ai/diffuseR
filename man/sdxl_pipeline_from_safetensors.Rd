% tinyrox says don't edit this manually, but it can't stop you!
\name{sdxl_pipeline_from_safetensors}
\alias{sdxl_pipeline_from_safetensors}
\title{Assemble a native SDXL pipeline from a diffusers safetensors directory}
\usage{
sdxl_pipeline_from_safetensors(diffusers_dir, devices = NULL,
                               unet_dtype = NULL, verbose = TRUE)
}
\arguments{
\item{diffusers_dir}{Directory with \code{unet/}, \code{vae/},
\code{text_encoder/}, \code{text_encoder_2/} subdirectories.}

\item{devices}{Named list of component devices (\code{unet},
\code{decoder}, \code{text_encoder}, \code{text_encoder2});
defaults to all-CPU. \code{text_encoder2} defaults to the
\code{text_encoder} device when unset.}

\item{unet_dtype}{A torch dtype for the UNet (default float16 on CUDA,
float32 on CPU).}

\item{verbose}{Logical.}
}
\value{
A list with \code{unet}, \code{decoder}, \code{text_encoder},
  \code{text_encoder2}, \code{vae_scaling}, and \code{native_decode}
  (\code{TRUE}; the decoder already applies \code{post_quant_conv}).
}
\description{
Builds the two native CLIP text encoders, the native SDXL UNet, and the
VAE decode module from a diffusers directory using the
\code{*_from_safetensors} constructors, places each on its component
device, and returns the list the \code{\link{txt2img_sdxl}} denoise
loop expects.
}
\details{
Both encoders return their penultimate hidden state (SDXL feeds the
UNet the concatenated \code{[text_encoder (768) | text_encoder_2
(1280)]} = 2048-dim penultimate embeds); \code{text_encoder} uses
\code{quick_gelu} (OpenAI CLIP ViT-L) and \code{text_encoder_2} uses
exact GELU (OpenCLIP bigG). The pooled \code{text_embeds} come from
\code{text_encoder_2}'s full stack. The VAE decodes in float32 (the
SDXL fp16 VAE overflows in fp16) and its \code{scaling_factor} is read
from \code{vae/config.json}.

}
