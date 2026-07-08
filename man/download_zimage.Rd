% tinyrox says don't edit this manually, but it can't stop you!
\name{download_zimage}
\alias{download_zimage}
\title{Download and Prepare Z-Image-Turbo Weights}
\description{
Downloads Z-Image-Turbo from HuggingFace (Apache-2.0, ungated) and
quantizes the 6B transformer to a local fp8 (~6.3 GB) or NF4
(~3.6 GB) artifact. The checkpoint ships the transformer in float32
(24.6 GB), so the one-time quantize saves a lot of disk and load
time.
}
