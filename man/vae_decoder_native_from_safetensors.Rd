% tinyrox says don't edit this manually, but it can't stop you!
\name{vae_decoder_native_from_safetensors}
\alias{vae_decoder_native_from_safetensors}
\title{Build a native VAE decoder from a diffusers safetensors directory}
\usage{
vae_decoder_native_from_safetensors(path, latent_channels = 4L, verbose = TRUE,
                                    ...)
}
\arguments{
\item{path}{Path to the VAE directory (containing
\code{diffusion_pytorch_model.safetensors}) or the file itself.}

\item{latent_channels}{Latent channel count (4 for SD/SDXL, 16 for FLUX).}

\item{verbose}{Print how many parameters were loaded.}

\item{...}{Overrides for \code{\link{vae_decoder_native}} constructor args.}
}
\value{
The native VAE decoder in eval mode.
}
\description{
The safetensors counterpart to the TorchScript decoder path:
constructs \code{\link{vae_decoder_native}} and loads the decoder half
of a diffusers AutoencoderKL checkpoint (no TorchScript, so it works
on Blackwell). \code{latent_channels} defaults to 4 (SD/SDXL); pass 16
for the FLUX/SD3 VAE. The SD/SDXL and FLUX VAEs share the decoder
shape and differ only in that channel count.
}
