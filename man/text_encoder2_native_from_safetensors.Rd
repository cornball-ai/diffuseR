% tinyrox says don't edit this manually, but it can't stop you!
\name{text_encoder2_native_from_safetensors}
\alias{text_encoder2_native_from_safetensors}
\title{Build a native SDXL text encoder 2 from a diffusers safetensors directory}
\usage{
text_encoder2_native_from_safetensors(path, return_penultimate = TRUE,
                                      verbose = TRUE, ...)
}
\arguments{
\item{path}{diffusers text_encoder_2 directory (config.json +
model.safetensors) or the config.json path.}

\item{return_penultimate}{Return the penultimate hidden state for the
SDXL cross-attention embeds (default TRUE); the pooled output is
always computed from the full stack.}

\item{verbose}{Print how many parameters were loaded.}

\item{...}{Overrides for \code{\link{text_encoder2_native}} args.}
}
\value{
The native text encoder 2 in eval mode.
}
\description{
Reads the CLIPTextConfig from \code{<dir>/config.json}, constructs
\code{\link{text_encoder2_native}} to match, and loads
\code{model.safetensors} - the safetensors counterpart to the
TorchScript \code{text_encoder_2} path (no TorchScript, Blackwell-safe).
}
