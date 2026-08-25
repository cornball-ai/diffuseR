% tinyrox says don't edit this manually, but it can't stop you!
\name{download_prebuilt}
\alias{download_prebuilt}
\title{Hosted prebuilt NF4 artifacts}
\description{
diffuseR hosts prebuilt NF4 artifacts for the two models whose
licenses permit redistribution: FLUX.2-klein-4B and Z-Image-Turbo
(both Apache-2.0, ungated). The artifacts are the exact output of
\code{\link{flux_quantize}} (sub-2 GB shards, bf16 residents), so even
safetensors older than 0.3.0 reads them. Nothing else in the catalog is
hosted: FLUX.1-schnell sits behind a HuggingFace license gate and
LTX-2.3's license does not permit redistribution, so both still
download their sources and quantize locally.
}
\keyword{internal}
