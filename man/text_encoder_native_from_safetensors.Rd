% tinyrox says don't edit this manually, but it can't stop you!
\name{text_encoder_native_from_safetensors}
\alias{text_encoder_native_from_safetensors}
\title{Build a native CLIP text encoder from a diffusers safetensors directory}
\usage{
text_encoder_native_from_safetensors(path, apply_final_ln = TRUE,
                                     verbose = TRUE, ...)
}
\arguments{
\item{path}{diffusers text_encoder directory (config.json +
model.safetensors) or the config.json path.}

\item{apply_final_ln}{Apply the final layer norm in forward (default TRUE).}

\item{verbose}{Print how many parameters were loaded.}

\item{...}{Overrides for \code{\link{text_encoder_native}} args (e.g.
\code{gelu_type}).}
}
\value{
The native text encoder in eval mode.
}
\description{
Reads the CLIPTextConfig from \code{<dir>/config.json}, constructs
\code{\link{text_encoder_native}} to match, and loads
\code{model.safetensors} - the safetensors counterpart to the
TorchScript text-encoder path (no TorchScript, Blackwell-safe).
Handles SD21's OpenCLIP ViT-H and SDXL's CLIP ViT-L (which is the same
checkpoint as FLUX's \code{text_encoder}). \code{apply_final_ln}
governs only the forward output; the \code{final_layer_norm} weights
load either way. Use \code{TRUE} for SD21 and pooled CLIP outputs,
\code{FALSE} for the SDXL penultimate-layer prompt embeds.
}
