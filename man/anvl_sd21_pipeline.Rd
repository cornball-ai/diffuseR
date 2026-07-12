% tinyrox says don't edit this manually, but it can't stop you!
\name{anvl_sd21_pipeline}
\alias{anvl_sd21_pipeline}
\title{Stable Diffusion 2.1 end-to-end text-to-image pipeline (anvl)}
\description{
Wires the three already-ported anvl components into the full SD 2.1
text-to-image path: the CLIP text encoder (\code{\link{yq_clip_encoder}},
conditional + unconditional), a host-side DDIM scheduler
(\code{\link{yq_sd21_ddim_sigmas}} / \code{\link{yq_sd21_ddim_step}}),
the UNet (\code{\link{yq_sd_unet}}) run under classifier-free guidance,
and the VAE decoder (\code{\link{yq_sd_vae_decode}}). Mirrors
\code{diffuseR::txt2img_sd21} on the native-safetensors path
(\code{sd_pipeline_from_safetensors}): scaled-linear beta schedule,
v-prediction DDIM (eta 0), CFG combine
\code{uncond + scale * (cond - uncond)}, latent init scale 1.0, and the
0.18215 VAE latent rescale before decode.
}
\details{
Only the UNet and CLIP forwards are anvl (jit-compiled once, called per
step); the DDIM alpha schedule and per-step coefficients are plain R
scalars, exactly as the FlowMatch sigmas were for the FLUX ports.

}
