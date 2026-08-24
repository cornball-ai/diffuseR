% tinyrox says don't edit this manually, but it can't stop you!
\name{sd21_load_pipeline}
\alias{sd21_load_pipeline}
\title{Load the SD 2.1 pipeline in the family-loader convention}
\usage{
sd21_load_pipeline(
  model_dir = NULL,
  device = "cuda",
  unet_dtype = NULL,
  phase_offload = FALSE,
  verbose = TRUE
)
}
\arguments{
\item{model_dir}{Diffusers directory (with \code{unet/}, \code{vae/},
\code{text_encoder/}). NULL, the default, resolves the
\code{\link{download_sd21}} cache, fetching it if absent.}

\item{device}{Where the pipeline will compute once activated. Components
are built on the CPU regardless, because residency pins them there and
\code{\link{resident_activate}} moves them.}

\item{unet_dtype}{A torch dtype for the UNet. NULL means float32. Pass
\code{torch::torch_float16()} only if you have reason to believe the
overflow above no longer applies.}

\item{phase_offload}{Kept for signature parity with the other family
loaders. SD 2.1 has no phased path, so anything but FALSE is ignored.}

\item{verbose}{Logical.}
}
\value{
The list from \code{\link{sd_pipeline_from_safetensors}}, plus
  \code{phase_offload}.
}
\description{
The adapter \code{\link{resident_load}} needs, the counterpart to
\code{\link{sdxl_load_pipeline}}.
\code{\link{sd_pipeline_from_safetensors}} takes a required
\code{diffusers_dir} and a plural \code{devices} list; every family
loader takes an optional model directory and a singular \code{device}.
}
\details{
\strong{The UNet defaults to float32, not float16.} This is the opposite
of \code{\link{sdxl_load_pipeline}} and it is not a preference: SD 2.1's
attention overflows in float16 and the pipeline returns all-NaN, so a
float16 resident would load, activate, generate and hand back a blank
image without raising anything. \code{\link{txt2img_sd21}} already
defaults its native path to float32 for this reason; the resident loader
has to make the same choice, because it fixes the dtype before the
weights are pinned and nothing downstream can undo it.

Only the UNet goes to the card, as for SDXL, but for a different reason.
SDXL is undone by its float32 VAE decode; SD 2.1 never reaches the
decode. The float32 requirement above makes the denoise itself expensive
-- measured 11.49 GB allocated plus 3.0 GB of allocator slack at the
768x768 default, which does not fit a 15.47 GiB card with the weights
resident as well. That is not a residency artifact: a plain
\code{\link{txt2img_sd21}} with the same devices and dtype OOMs at
11.494 GB, within 0.04 GB of the resident run.

\code{\link{auto_devices}} already answers "unet on cuda, the rest on
cpu" for this model at every strategy, so residency is agreeing with the
package rather than inventing a placement.

}
\examples{
\dontrun{
res <- resident_load("sd21")
resident_activate(res)
img <- resident_generate(res, "a cat in a spacesuit", seed = 7)
resident_deactivate(res)
}

}
\seealso{
\code{\link{resident_load}}, \code{\link{sdxl_load_pipeline}}
}
