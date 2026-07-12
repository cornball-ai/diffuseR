% tinyrox says don't edit this manually, but it can't stop you!
\name{anvl_zimage_pipeline}
\alias{anvl_zimage_pipeline}
\title{Z-Image-Turbo end-to-end text-to-image pipeline (anvl / XLA backend)}
\description{
anvl re-implementation of \code{diffuseR::txt2img_zimage}: Qwen3-4B
prompt encoding (penultimate hidden state) feeds the caption
conditioning, a guidance-free FlowMatch Euler loop denoises over the
native \code{\link{yq_zimage_dit}} at the real config, and the FLUX.1
16-channel decoder (\code{\link{yq_zimage_vae_decode}}) renders pixels.
}
\details{
Everything runs f32 (anvl has no fp8/bf16): the quantized transformer
is dequantized to f32 on load (per-tensor \code{weight_scale}), the
Qwen3 encoder loads bf16-upcast-to-f32, so this matches a CPU-f32
torch reference run (fp8 dequant + f32 compute) rather than the
production bf16 GPU path.

}
