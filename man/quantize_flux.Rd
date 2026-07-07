% tinyrox says don't edit this manually, but it can't stop you!
\name{quantize_flux}
\alias{quantize_flux}
\title{FLUX Transformer Quantization and Loading}
\description{
Quantize the 12B FLUX transformer to NF4 (~7 GB, GPU-resident on
16 GB cards) or fp8 (~12 GB, CPU-resident and streamed per forward),
and load any format back into \code{\link{flux_transformer}}. Reuses
the LTX-2.3 quantization machinery (\code{ltx23_nf4_quantize},
\code{ltx23_nf4_linear}, \code{ltx23_fp8_linear}); only the cast set
and the diffusers directory layout are FLUX-specific.
}
