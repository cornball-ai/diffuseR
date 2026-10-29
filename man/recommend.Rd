% tinyrox says don't edit this manually, but it can't stop you!
\name{recommend}
\alias{recommend}
\title{Recommend a precision and device configuration for a model}
\usage{
recommend(
  model = c("sd21", "sdxl", "flux1", "flux2", "zimage", "ltx", "ltx25"),
  vram_gb = NULL,
  st_caps = NULL,
  host_ram_gb = NULL
)
}
\arguments{
\item{model}{"sd21", "sdxl", "flux1", "flux2", "zimage", "ltx"
(LTX-2.3), or "ltx25" (LTX-2.5).}

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
  \code{host_ram_gb}, \code{fork_suggested} (logical: the installed
  safetensors cannot read the tier the card could otherwise run), and
  \code{note} (the suggestion string, or NULL). For \code{"ltx25"} it
  also has \code{resident_gb}, \code{stream_text}, \code{split_text}
  and \code{resident_from} (see \code{\link{ltx23_memory_profile}}).
  The \code{fork_suggested} field name is
  historical -- the remedy used to be the cornball-ai fork and is now
  \code{install.packages("safetensors")}, since the fixes reached CRAN
  in 0.3.0 -- and is kept because it is part of the returned contract.
}
\description{
One VRAM-and-capability-aware recommendation for every diffuseR
model. The policy:
}
\details{
\itemize{
\item nf4 is the default tier for the quantized families (flux1,
flux2, zimage, ltx). Its weights are packed uint8 plus float32
blocks in sub-2 GB shards, which every safetensors reads, so it
always loads. The SD models ship no quantized weights; their
floor is fp16 with placement varying by VRAM.
\item When the card has room for a higher-quality tier (fp8 or
bf16) AND the installed safetensors can \emph{read} that dtype
(\code{\link{.st_can_read}}), that tier is recommended instead.
\item When the card has room but safetensors cannot read the tier,
nf4 is recommended and the upgrade suggestion is surfaced in
\code{note} (never an error).
}

This is the policy engine; it does no disk I/O and does not know which
artifacts are built. Loaders reconcile the recommendation with what is
on disk (see \code{\link{flux_load_pipeline}}). Thresholds are
validated on an RTX 5060 Ti (16 GB) and are deliberately conservative
elsewhere. Video sizing for \code{"ltx"} is coarse here; the LTX
pipeline uses \code{\link{ltx23_memory_profile}} for frame-aware
placement.

\code{"ltx"} is LTX-2.3, which has fp8 and nf4 artifacts: 10-14 GB
cards get fp8 streamed per linear. \code{"ltx25"} is LTX-2.5, which
has only the nf4 artifact: below 14 GB it keeps part of the nf4
transformer resident and streams the rest, and the result carries
the \code{\link{ltx23_memory_profile}} fields that configure that
(\code{resident_gb}, \code{stream_text}, \code{split_text},
\code{resident_from}). Its staging is always pinned on the GPU
tiers: block streaming from pageable memory is slow, and the
host-RAM fields already bound the footprint (measured at 768x512x121
on an 8 GB card: 20.1 GB high-water with resident host copies, 16.0
GB with \code{resident_from = "disk"}).

The pinning decision: phase-swapped weights are page-locked host
copies (see \code{\link{staging}}) that transfer at DMA rate -
but pinned pages are unswappable, so on small-RAM machines they turn
memory pressure into OOM kills. \code{pin} is TRUE when available
host RAM covers the model's pinned set twice over, FALSE below that,
FALSE on the cpu tier (nothing stages), and TRUE when RAM cannot be
detected (page-locking already fails soft per component). The LTX
pipeline, the Gemma3 encoder, and the FLUX-family image loaders
(flux1, flux2, zimage) take \code{pin} arguments and stage pinned
weights (see \code{\link{staging}}); the SD-family loaders place
components statically and do not phase-swap, so \code{pin} is inert
for them. \code{options(diffuseR.pin_staging)} is the global switch.

}
\examples{
# Stating vram_gb and st_caps makes the policy deterministic: no GPU
# and no installed safetensors needed.
r <- recommend("flux1", vram_gb = 16,
               st_caps = list(bfloat16 = TRUE, float8_e4m3fn = FALSE))
r$precision       # "nf4": fp8 fits the card, but cannot be read
r$fork_suggested  # TRUE
cat(r$note)       # the update-or-nf4 message

# Same card, once safetensors can read float8
recommend("flux1", vram_gb = 16,
          st_caps = list(bfloat16 = TRUE, float8_e4m3fn = TRUE))$precision

# Auto-detect VRAM and probe the installed safetensors
str(recommend("flux2"))

# LTX-2.5 on an 8 GB card with 24 GB of RAM: nf4, 3.6 GB resident,
# text phase split off
recommend("ltx25", vram_gb = 7.6, host_ram_gb = 24)[c("precision",
          "resident_gb", "split_text", "resident_from")]
}
