% tinyrox says don't edit this manually, but it can't stop you!
\name{resident_load}
\alias{resident_load}
\title{Load a diffusion pipeline as a resident handle}
\usage{
resident_load(
  model = c("flux2", "flux1", "flux3", "zimage", "ltx", "sdxl", "sd21"),
  device = "cuda",
  ...,
  verbose = TRUE,
  text_encoder = NULL,
  tokenizer = NULL,
  stream_text = FALSE
)
}
\arguments{
\item{model}{One of "flux1", "flux2", "flux3", "zimage", "ltx", "sdxl",
"sd21".}

\item{device}{Target CUDA device, e.g. "cuda" or "cuda:1".}

\item{...}{Passed to the family loader (\code{\link{flux_load_pipeline}},
\code{\link{flux2_load_pipeline}}, \code{\link{flux3_load_pipeline}},
\code{\link{zimage_load_pipeline}},
\code{\link{ltx23_load_pipeline}} or
\code{\link{sdxl_load_pipeline}}). \code{ltx} requires
\code{checkpoint_path}; \code{sdxl} needs nothing (it defaults to the
\code{\link{download_sdxl}} cache). \code{flux3} takes
\code{model_dir}, \code{text_dir} and \code{support_dir} to read
from given paths instead of the default locations.}

\item{verbose}{Print progress messages.}

\item{stream_text}{\code{ltx} with a Gemma 4 \code{text_encoder} only:
stream the encoder's layers to the card one at a time instead of
onloading it whole for the encode (\code{\link{load_gemma4_nf4}}'s
\code{stream}). \code{\link{recommend}("ltx25")} asks for it below
12 GB. On small cards pass \code{resident_gb} too (through \code{...}
to \code{\link{ltx23_load_pipeline}}), so the transformer keeps that
many GB on the card and streams the rest of its blocks.}

\item{text_encoder,tokenizer}{\code{ltx} only: paths to the Gemma3 encoder
artifact and the tokenizer directory, or for LTX-2.5 the Gemma 4 NF4
artifact (\code{\link{gemma4_quantize_nf4}}) for both, since it carries
its tokenizer. \code{\link{ltx23_load_pipeline}}
does not load these -- \code{\link{txt2vid_ltx2}} takes them per call --
so a handle built without them can be activated and cannot generate.
Given here they are loaded ONCE, pinned on the host, and passed to every
generate; the encoder rides to the card for its phase and back off, the
way the pipeline's own components do, so it is never resident beside the
transformer.}
}
\value{
A \code{diffuseR_resident} handle (an environment). Inspect it
  with \code{\link{resident_status}}; the fields of interest are the
  state, the bound device, the component names, and the pinned host
  byte count.
}
\description{
Loads a pipeline once and keeps its weights page-locked on the host
for the life of the handle. The GPU representation is created by
\code{\link{resident_activate}} and destroyed by
\code{\link{resident_deactivate}}, so a 16 GB card can hand itself
between models without either one re-reading its weights from disk.
}
\details{
The handle is bound to one explicit GPU at load: a bare \code{"cuda"}
resolves to the current device now, and every later transition uses
that index, so the handle cannot drift to whichever GPU happens to be
current at transition time.

One caveat on multi-GPU hosts: the family loader itself runs on the
\emph{current} device, and only the residency handle is bound to
\code{device}. Loading with \code{device = "cuda:1"} from a session
whose current device is 0 therefore stages through GPU 0 before the
first activation lands on GPU 1. Wrap the call in
\code{torch::with_device(device = "cuda:1", ...)} when that matters.

The pipeline is left \emph{inactive} (weights pinned on the host, no
VRAM held). Call \code{\link{resident_activate}} before generating.

For \code{ltx}, \code{stage_from = "disk"} (through \code{...}) keeps
no weights in host memory: the pipeline and a Gemma 4
\code{text_encoder} both read every onload from their memory-mapped
NF4 artifacts, and \code{pinned_bytes} counts only the small computed
buffers. It saves ~26 GB of host RAM for a re-read per phase (and per
step for the streamed blocks), which the OS page cache serves when it
can. A Gemma 3 \code{text_encoder} is refused.

}
\examples{
\dontrun{
res <- resident_load("flux2")
resident_activate(res)
img <- resident_generate(res, "a cat in a spacesuit", seed = 7)
resident_deactivate(res) # VRAM freed, weights stay pinned in RAM
resident_activate(res) # fast: DMA copy, no disk
resident_unload(res)
}

}
\seealso{
\code{\link{resident_activate}}, \code{\link{resident_status}}
}
