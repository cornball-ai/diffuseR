% tinyrox says don't edit this manually, but it can't stop you!
\name{anvl_sdxl_pipeline}
\alias{anvl_sdxl_pipeline}
\title{Stable Diffusion XL end-to-end text-to-image pipeline (anvl)}
\description{
Wires the ported anvl SDXL components into the full text-to-image path:
the dual CLIP text encoders (\code{\link{yq_sdxl_clip_encoders}}) for the
concatenated penultimate context and the bigG pooled embed, a host-side
DDIM scheduler (\code{\link{yq_sdxl_ddim_schedule}} /
\code{\link{yq_sdxl_ddim_step}}), the UNet (\code{\link{yq_sdxl_unet}})
run under classifier-free guidance with the SDXL added text-time
conditioning (pooled text embeds + sinusoidal time-ids), and the VAE
decoder (\code{\link{yq_sdxl_vae_decode}}). Mirrors
\code{diffuseR::txt2img_sdxl}: scaled-linear beta schedule, epsilon
DDIM (eta 0), CFG combine \code{uncond + scale * (cond - uncond)}, and
the \code{negative_prompt = NULL} convention of zeroed unconditional
embeddings.
}
\details{
\strong{Conditioning provenance.} The context is the concatenated
\emph{penultimate} CLIP-L + bigG hidden states (diffusers SDXL
clip-skip = None, \code{hidden_states[-2]}), matching the anvl encoder
design and the \code{anvl_test_sdxl_clip} parity fixture, not the
simplified full-forward call in \code{txt2img_sdxl}.

Only the UNet and CLIP forwards are anvl (jit-compiled once, the UNet
called twice per step); the DDIM alpha schedule and per-step
coefficients are plain R scalars, exactly as the SD 2.1 pipeline did.

}
