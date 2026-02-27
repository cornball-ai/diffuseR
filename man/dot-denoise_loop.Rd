% tinyrox says don't edit this manually, but it can't stop you!
\name{.denoise_loop}
\alias{.denoise_loop}
\title{Run a denoising loop}
\usage{
.denoise_loop(
  latents,
  dit,
  schedule,
  video_embeds,
  audio_embeds,
  audio_latents,
  latent_frames,
  latent_height,
  latent_width,
  dit_device,
  latent_dtype,
  fps,
  use_cfg,
  distilled,
  memory_profile,
  neg_video_embeds = NULL,
  neg_audio_embeds = NULL,
  guidance_scale = 1,
  verbose = TRUE,
  stage_label = NULL
)
}
\description{
Shared Euler-step loop used by both Stage 1 and Stage 2 of the pipeline.
}
\keyword{internal}
