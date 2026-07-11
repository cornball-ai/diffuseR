% tinyrox says don't edit this manually, but it can't stop you!
\name{load_text_encoder2_safetensors}
\alias{load_text_encoder2_safetensors}
\title{Load HF safetensors weights into the native SDXL text encoder 2}
\usage{
load_text_encoder2_safetensors(native_encoder, path, verbose = TRUE)
}
\arguments{
\item{native_encoder}{Native text encoder 2 module}

\item{path}{Path to model.safetensors (or a directory containing it)}

\item{verbose}{Print loading progress}
}
\value{
The native encoder with loaded weights (invisibly)
}
\description{
Loads a HuggingFace CLIPTextModelWithProjection \code{model.safetensors}
(SDXL's OpenCLIP ViT-bigG \code{text_encoder_2}) into
\code{\link{text_encoder2_native}}. Same key layout as encoder 1, plus
the top-level \code{text_projection.weight} that produces the pooled
\code{text_embeds}.
}
