% tinyrox says don't edit this manually, but it can't stop you!
\name{load_text_encoder_safetensors}
\alias{load_text_encoder_safetensors}
\title{Load HF safetensors weights into the native CLIP text encoder}
\usage{
load_text_encoder_safetensors(native_encoder, path, verbose = TRUE)
}
\arguments{
\item{native_encoder}{Native text encoder module}

\item{path}{Path to model.safetensors (or a directory containing it)}

\item{verbose}{Print loading progress}
}
\value{
The native encoder with loaded weights (invisibly)
}
\description{
Loads a HuggingFace CLIPTextModel \code{model.safetensors} (e.g.
FLUX.1-schnell's \code{text_encoder}) into
\code{\link{text_encoder_native}}, reusing the TorchScript key remaps
minus the export prefixes.
}
