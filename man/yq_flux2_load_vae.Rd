% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_flux2_load_vae}
\alias{yq_flux2_load_vae}
\title{Load FLUX.2 VAE decoder weights into an anvl pytree}
\usage{
yq_flux2_load_vae(path, device = "cpu")
}
\arguments{
\item{path}{Path to \code{vae/diffusion_pytorch_model.safetensors}.}

\item{device}{Character. Target device.}
}
\value{
VAE weights pytree; \code{bn_mean}/\code{bn_var} are returned
  as R vectors for host-side de-normalization
  (\code{\link{yq_flux2_vae_prepare}}).
}
\description{
Reads the decoder half of the \code{AutoencoderKLFlux2} checkpoint
(bf16 upcast to f32): \code{post_quant_conv}, the decoder body, the
output head, and the BatchNorm latent statistics. Conv weights are
loaded in torch \code{[out, in, kH, kW]} layout (what
\code{nv_conv2d} expects); the attention linears are transposed to
\code{[in, out]} for \code{yq_linear}. Encoder / quant_conv keys are
skipped (txt2img needs only the decode path).
}
