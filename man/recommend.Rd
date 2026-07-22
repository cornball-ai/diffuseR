% tinyrox says don't edit this manually, but it can't stop you!
\name{recommend}
\alias{recommend}
\title{Recommend a precision and device configuration for a model}
\usage{
recommend(model = c("sd21", "sdxl", "flux1", "flux2", "zimage", "ltx"),
          vram_gb = NULL, st_caps = NULL, host_ram_gb = NULL)
}
\arguments{
\item{model}{"sd21", "sdxl", "flux1", "flux2", "zimage", or "ltx".}

\item{vram_gb}{Numeric or NULL. Free VRAM in GB; auto-detected via
nvidia-smi when NULL.}

\item{st_caps}{NULL or a named logical list with \code{bfloat16}
and/or \code{float8_e4m3fn} - the safetensors READ capabilities.
NULL probes the installed safetensors.}

\item{host_ram_gb}{Numeric or NULL. Available host RAM in GB;
auto-detected (Linux \code{MemAvailable}) when NULL, NA where
undetectable.}
}
\value{
A list with \code{model}, \code{precision}, \code{devices}
  (named component -> device map), \code{offload} (phase-offloading
  logical), \code{max_pixels}, \code{text_device}, \code{attn_chunk},
  \code{vram_gb}, \code{pin} (page-lock the phase-swapped host
  copies), \code{pinned_set_gb} (estimated pinned bytes),
  \code{host_ram_gb}, \code{fork_suggested} (logical), and
  \code{note} (the fork suggestion string, or NULL).
}
\description{
One VRAM-and-capability-aware recommendation for every diffuseR
model. The policy:
}
\details{
\itemize{
\item nf4 is the default tier. Its weights are packed uint8 plus
float32 blocks in sub-2 GB shards, which every safetensors reads,
so it always loads.
\item When the card has room for a higher-quality tier (fp8 or
bf16) AND the installed safetensors can \emph{read} that dtype
(\code{\link{.st_can_read}}), that tier is recommended instead.
\item When the card has room but safetensors cannot read the tier,
nf4 is recommended and the fork suggestion is surfaced in
\code{note} (never an error).
}

This is the policy engine; it does no disk I/O and does not know which
artifacts are built. Loaders reconcile the recommendation with what is
on disk (see \code{\link{flux_load_pipeline}}). Thresholds are
validated on an RTX 5060 Ti (16 GB) and are deliberately conservative
elsewhere. Video sizing for \code{"ltx"} is coarse here; the LTX
pipeline uses \code{\link{ltx23_memory_profile}} for frame-aware
placement.

The pinning decision: phase-swapped weights are page-locked host
copies (see \code{\link{staging_ltx23}}) that transfer at DMA rate -
but pinned pages are unswappable, so on small-RAM machines they turn
memory pressure into OOM kills. \code{pin} is TRUE when available
host RAM covers the model's pinned set twice over, FALSE below that,
FALSE on the cpu tier (nothing stages), and TRUE when RAM cannot be
detected (page-locking already fails soft per component). Today the
LTX pipeline and Gemma3 encoder loaders take \code{pin} arguments
and stage pinned weights (see \code{\link{staging_ltx23}}); the
image-model loaders do not stage weights yet, so for them \code{pin}
is forward-looking policy with no consumer.
\code{options(diffuseR.pin_staging)} is the global switch.

}
\examples{
\dontrun{
# Auto-detect VRAM and probe the installed safetensors
recommend("flux2")

# A 16 GB card without float8 support: fp8 wanted, nf4 recommended
r <- recommend("flux1", vram_gb = 16,
               st_caps = list(bfloat16 = TRUE, float8_e4m3fn = FALSE))
r$precision       # "nf4"
r$fork_suggested  # TRUE
cat(r$note)       # the fork-or-nf4 message
}
}
