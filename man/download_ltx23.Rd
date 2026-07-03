% tinyrox says don't edit this manually, but it can't stop you!
\name{download_ltx23}
\alias{download_ltx23}
\title{Download and Prepare LTX-2.3 Model Weights}
\description{
Downloads the LTX-2.3 distilled checkpoint (46 GB, LTX-2 Community
License) and the Gemma3 text encoder from HuggingFace with an explicit
consent prompt, then quantizes the transformer to the local fp8
artifact (~26 GB) used by the GPU-poor pipeline.
}
