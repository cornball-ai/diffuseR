% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx25_open_checkpoint}
\alias{ltx25_open_checkpoint}
\title{Open the LTX-2.5 component files as one checkpoint}
\usage{
ltx25_open_checkpoint(transformer, text_encoder, video_vae, audio_vae = NULL)
}
\arguments{
\item{transformer}{The 2.5 transformer file
(\code{ltx-2.5-22b-distilled-transformer-bf16.safetensors}) or an
fp8/NF4 artifact directory quantized from it.}

\item{text_encoder}{The 2.5 Gemma 4 text encoder file, or its NF4
artifact directory (\code{\link{gemma4_quantize_nf4}} keeps the
projection); only \code{text_embedding_projection.*} is used here.}

\item{video_vae}{The 2.5 video VAE file. Use the convolutional decoder
(\code{ltx-2.5-video-vae-conv-bf16.safetensors}); the default
diffusion decoder is not supported.}

\item{audio_vae}{Optional 2.5 audio VAE + vocoder file
(\code{ltx-2.5-audio-vae-bf16.safetensors}).}
}
\value{
An \code{ltx23_checkpoint} whose \code{handle$get_tensor()}
  reads each key from its source file.
}
\description{
LTX-2.3 shipped every component in one file; LTX-2.5 splits them: the
transformer (with the text connectors), the Gemma 4 text encoder file
(which carries the connectors' \code{text_embedding_projection}), the
video VAE, and the audio VAE plus vocoder. This presents them as a
single \code{ltx23_checkpoint} under the 2.3 key names (the video VAE
keys gain their \code{vae.} prefix), so \code{\link{ltx23_load_pipeline}}
and its component loaders work unchanged.
}
