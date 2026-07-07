% tinyrox says don't edit this manually, but it can't stop you!
\name{vae_decoder_native}
\alias{vae_decoder_native}
\title{Native VAE Decoder}
\usage{
vae_decoder_native(latent_channels = 4, out_channels = 3,
                   block_channels = c(512, 512, 256, 128), norm_groups = 32)
}
\arguments{
\item{latent_channels}{Number of latent channels (4 for SD/SDXL,
16 for FLUX/SD3)}

\item{out_channels}{Number of output channels (default 3 for RGB)}

\item{block_channels}{Decoder block channels (reversed encoder
block_out_channels; default matches SD/SDXL and FLUX)}

\item{norm_groups}{Group norm groups (default 32; must divide every
entry of \code{block_channels})}
}
\value{
An nn_module representing the VAE decoder
}
\description{
Native R torch implementation of the SDXL VAE decoder.
Replaces TorchScript decoder for better GPU compatibility.
}
\examples{
\dontrun{
decoder <- vae_decoder_native()
load_decoder_weights(decoder, "path/to/decoder.pt")
latents <- torch::torch_randn(c(1, 4, 64, 64))
image <- decoder(latents)
}
}
