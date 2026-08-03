% tinyrox says don't edit this manually, but it can't stop you!
\name{resident_load}
\alias{resident_load}
\title{Load a diffusion pipeline as a resident handle}
\usage{
resident_load(model = c("flux2", "flux1", "zimage", "ltx"), device = "cuda",
              ..., verbose = TRUE)
}
\arguments{
\item{model}{One of "flux1", "flux2", "zimage", "ltx".}

\item{device}{Target CUDA device, e.g. "cuda" or "cuda:1".}

\item{...}{Passed to the family loader (\code{\link{flux_load_pipeline}},
\code{\link{flux2_load_pipeline}}, \code{\link{zimage_load_pipeline}}
or \code{\link{ltx23_load_pipeline}}). \code{ltx} requires
\code{checkpoint_path}.}

\item{verbose}{Print progress messages.}
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
