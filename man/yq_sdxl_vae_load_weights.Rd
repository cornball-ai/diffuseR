% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_sdxl_vae_load_weights}
\alias{yq_sdxl_vae_load_weights}
\title{Load SDXL VAE decoder weights into an anvl pytree}
\usage{
yq_sdxl_vae_load_weights(path, device = "cpu", strict = TRUE)
}
\arguments{
\item{path}{Path to \code{vae/diffusion_pytorch_model.safetensors}.}

\item{device}{Character. Target device.}

\item{strict}{Logical. Enforce the exact decode-key census.}
}
\value{
Weights pytree for \code{\link{yq_sdxl_vae_decode}}.
}
\description{
Reads the decode half of the diffusers SDXL \code{AutoencoderKL}
checkpoint (F16 upcast to f32): \code{post_quant_conv}, the decoder
body, and the output head. Conv weights load raw \code{[out, in, kH,
kW]} (what \code{\link[anvl]{nv_conv2d}} expects); the attention
linears transpose to \code{[in, out]} for \code{\link[yunque]{yq_linear}}.
Encoder and \code{quant_conv} keys are skipped (txt2img needs only the
decode path). With \code{strict = TRUE} every \code{decoder.*} /
\code{post_quant_conv.*} key must be consumed exactly once (and no
requested key may be absent), so a wrong architecture or a missed
weight fails loudly. The SDXL VAE key tree is byte-for-byte the SD 2.1
one, so this loader mirrors \code{diffuseR}'s SD 2.1 VAE loader.
}
