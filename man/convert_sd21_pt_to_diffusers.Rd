% tinyrox says don't edit this manually, but it can't stop you!
\name{convert_sd21_pt_to_diffusers}
\alias{convert_sd21_pt_to_diffusers}
\title{Convert cornball SD 2.1 TorchScript weights to a diffusers artifact}
\usage{
convert_sd21_pt_to_diffusers(pt_dir = NULL, output_dir = NULL,
                             dtype = c("float16", "float32"), verbose = TRUE)
}
\arguments{
\item{pt_dir}{Directory holding \code{unet-cpu.pt},
\code{decoder-cpu.pt}, \code{text_encoder-cpu.pt} (default: the
diffuseR \code{sd21} data location).}

\item{output_dir}{Output diffusers directory (default: the
\code{sd_pipeline_from_safetensors} / \code{download_sd21} location).}

\item{dtype}{\code{"float16"} (default, the hosted tier) or
\code{"float32"}.}

\item{verbose}{Logical.}
}
\value{
Invisibly, \code{output_dir}.
}
\description{
Rebuilds a diffusers-layout directory (\code{unet/}, \code{vae/},
\code{text_encoder/}) from the cornball-ai/sd21-R TorchScript
component \code{.pt} files, so the native safetensors pipeline
(\code{\link{download_sd21}} / \code{\link{sd_pipeline_from_safetensors}})
can load SD 2.1 with no TorchScript.
}
\details{
A TorchScript trace preserves the exact parameter tensors, so the
result is bit-identical to the source at the chosen dtype. This is the
provenance-clean way to build the hosted artifact: the upstream
\code{stabilityai/stable-diffusion-2-1} repo was deprecated, SD 2.1 is
CreativeML OpenRAIL++-M (redistributable), and cornball already hosts
these weights as \code{.pt}. At \code{float16} the components are all
sub-2 GB single files (unet ~1.7 GB, text_encoder ~0.65 GB, vae
~0.16 GB), so they load on stock CRAN safetensors.

}
