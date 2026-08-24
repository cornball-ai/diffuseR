% tinyrox says don't edit this manually, but it can't stop you!
\name{sdxl_load_pipeline}
\alias{sdxl_load_pipeline}
\title{Load the SDXL pipeline in the family-loader convention}
\usage{
sdxl_load_pipeline(
  model_dir = NULL,
  device = "cuda",
  unet_dtype = NULL,
  phase_offload = FALSE,
  verbose = TRUE
)
}
\arguments{
\item{model_dir}{Diffusers directory (with \code{unet/}, \code{vae/},
\code{text_encoder/}, \code{text_encoder_2/}). NULL, the default,
resolves the \code{\link{download_sdxl}} cache, fetching it if absent.}

\item{device}{Where the pipeline will COMPUTE once activated. Components
are built on the CPU regardless, because residency pins them there and
\code{\link{resident_activate}} moves them; this only picks the dtype.}

\item{unet_dtype}{A torch dtype for the UNet. NULL picks float16 for a
CUDA device and float32 for CPU.}

\item{phase_offload}{Kept for signature parity with the other family
loaders. SDXL has no phased path, so anything but FALSE is ignored.}

\item{verbose}{Logical.}
}
\value{
The list from \code{\link{sdxl_pipeline_from_safetensors}}, plus
  \code{phase_offload}.
}
\description{
The adapter \code{\link{resident_load}} needs.
\code{\link{sdxl_pipeline_from_safetensors}} predates the residency layer
and has its own signature: a required \code{diffusers_dir} and a plural
\code{devices} list, where every other family loader takes an optional
model directory and a singular \code{device}. This translates.
}
\details{
Two choices are not cosmetic:

\code{unet_dtype} is fixed HERE rather than at generation time.
\code{sdxl_pipeline_from_safetensors} defaults it from the component
device, so loading to CPU for pinning would page-lock a float32 UNet
(~10 GB) and then render in float32. A resident handle must decide the
dtype from where it will COMPUTE, not from where the weights are parked
while pinned.

\code{phase_offload} is FALSE, unlike every other family. SDXL has no
per-phase offloading path -- \code{\link{txt2img_sdxl}} places components
once and leaves them -- so activation has to be a real transfer rather
than an ownership claim. \code{\link{resident_activate}} reads this field
off the pipeline and does the right thing.

The pipeline also carries \code{gpu_components = "unet"}: all four
components are pinned, but only the UNet is put on the card, and the text
encode and VAE decode run on the host. The 8.0 GB pinned set makes
onloading everything look affordable on a 16 GB card, and it is not --
SDXL decodes 1024x1024 in float32 and that peak lands while the UNet is
still resident, which reached 14.38 GiB of 15.47 GiB and OOMed. On the
12 GB cards this wrapper exists for, only the UNet was ever going to fit.

}
\examples{
\dontrun{
res <- resident_load("sdxl")
resident_activate(res)
img <- resident_generate(res, "a cat in a spacesuit", seed = 7)
resident_deactivate(res)
}

}
\seealso{
\code{\link{resident_load}}
}
