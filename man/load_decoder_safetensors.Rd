% tinyrox says don't edit this manually, but it can't stop you!
\name{load_decoder_safetensors}
\alias{load_decoder_safetensors}
\title{Load HF safetensors VAE weights into the native decoder}
\usage{
load_decoder_safetensors(native_decoder, path, verbose = TRUE)
}
\arguments{
\item{native_decoder}{Native VAE decoder module}

\item{path}{Path to the VAE .safetensors file (or a directory
containing diffusion_pytorch_model.safetensors)}

\item{verbose}{Print loading progress}
}
\value{
The native decoder with loaded weights (invisibly)
}
\description{
Loads the decoder half of a diffusers AutoencoderKL safetensors file
(e.g. FLUX.1-schnell's \code{vae/diffusion_pytorch_model.safetensors}).
Keys under \code{decoder.} map to the native module 1:1; encoder and
quant-conv keys are skipped (the FLUX VAE has no quant convs, and
txt2img needs no encoder).
}
