% tinyrox says don't edit this manually, but it can't stop you!
\name{txt2vid_ltx2}
\alias{txt2vid_ltx2}
\title{Generate video (and audio) with LTX-2.3}
\usage{
txt2vid_ltx2(prompt, pipeline, text_encoder = NULL, tokenizer = NULL,
             prompt_embeds = NULL, connector_embeds = NULL, width = 768L,
             height = 512L, num_frames = 121L, frame_rate = 24,
             sigmas = ltx23_distilled_sigmas(), guidance_scale = 1,
             seed = NULL, device = "cuda", dtype = "bfloat16", filename = NULL,
             max_sequence_length = 1024L, decode_video = TRUE,
             decode_audio = TRUE, two_stage = FALSE, upsampler = NULL,
             adain_factor = 1, tone_map_compression = 0, phase_offload = TRUE,
             image = NULL, condition_video = NULL, conditioning_frames = 9L,
             cond_noise_scale = 0, condition_latents = NULL,
             resident = character(), trim_frames = 0L, audio = NULL,
             verbose = TRUE)
}
\arguments{
\item{prompt}{Character. The prompt.}

\item{pipeline}{An \code{ltx23_pipeline} from
\code{\link{ltx23_load_pipeline}}.}

\item{prompt_embeds}{Optional precomputed list with
\code{prompt_embeds} (raw stacked Gemma3 states) and
\code{prompt_attention_mask}; bypasses the text encoder.}

\item{connector_embeds}{Optional precomputed text-connector outputs:
a list with \code{video_text_embedding},
\code{audio_text_embedding}, and \code{attention_mask} (the result
of \code{pipeline$connectors} on the Gemma3 states). The prompt is
constant across a chained track, so compute this once and pass it
to every chunk: it skips the per-call connectors phase, whose
GPU-side handling of the raw hidden-state stack does not fit next
to a resident transformer.}

\item{num_frames}{Integer. 8k + 1 frames (e.g. 121).}

\item{frame_rate}{Numeric. Frames per second.}

\item{sigmas}{Numeric vector. Denoising schedule (default: official
distilled schedule; must end in 0).}

\item{guidance_scale}{Numeric. Only 1 (no CFG) is supported; the
distilled checkpoints are trained for CFG-free sampling.}

\item{seed}{Integer or NULL.}

\item{device}{Character. Compute device for the denoising loop.}

\item{dtype}{Character. Model compute dtype ("bfloat16" or "float32").}

\item{filename}{Character or NULL. Output video path (.mp4). Audio is
muxed in when the av package is available.}

\item{max_sequence_length}{Integer. Text token length (multiple of 128).}

\item{two_stage}{Logical. Generate at half resolution, upsample the
latents 2x spatially, and refine over the stage-2 schedule
(resolution must then be a multiple of 64; requires
\code{upsampler}).}

\item{upsampler}{An \code{\link{ltx23_latent_upsampler}} (see
\code{\link{ltx23_load_upsampler}}).}

\item{adain_factor}{Numeric. AdaIN blend of the upsampled latents
toward the stage-1 statistics (0 disables).}

\item{tone_map_compression}{Numeric in [0, 1]. Optional latent tone
mapping before stage 2.}

\item{phase_offload}{Logical. Move each small component to the compute
device only for its phase (text encoding, upsampling, decoding) and
back to the CPU afterwards, keeping the denoise phase as the sole
GPU tenant.}

\item{image}{Optional start image for image-to-video: a PNG/JPEG path
or an [H, W, 3] array in [0, 1]. The image conditions the first
frame; the rest of the video is generated (reference i2v).}

\item{condition_video}{Optional continuation source: a video path
(its trailing \code{conditioning_frames} frames are used) or an
[F, H, W, 3] array. The clip's tail becomes the frozen prefix of
the new video, so the output's first \code{conditioning_frames}
frames overlap the source (trim or crossfade when concatenating).}

\item{conditioning_frames}{Integer. Trailing pixel frames taken from
\code{condition_video} (8k + 1, default 9 = 2 latent frames).}

\item{cond_noise_scale}{Numeric in [0, 1]. Optional partial noising
of the conditioned tokens (0 = keep them exactly).}

\item{condition_latents}{Optional continuation source already in
latent space: normalized video latents [1, 128, k, height/32,
width/32] (e.g. \code{\link{ltx23_tail_latents}} on a previous
result), used directly as the frozen prefix with no VAE encode.
Mutually exclusive with \code{image} and \code{condition_video}.}

\item{resident}{Character vector of pipeline component names
("transformer", "vae", "audio_vae", "connectors", "vocoder") to
keep on the compute device after their phase instead of
offloading, for callers running several generations back to back
(chained chunks). Components already on the device are not
re-copied on later calls.}

\item{trim_frames}{Integer. Drop this many leading pixel frames
from the decoded video (and the saved file), e.g. the
conditioning-head overlap of a continuation. The returned
\code{latents} keep the full sequence (tail slicing for chaining
needs it). Audio is muxed exactly as supplied, so drop any head
padding from the conditioning audio when trimming.}

\item{audio}{Optional conditioning audio for audio-driven generation
(lip sync): a file path (decoded via \code{av}) or a matrix
[2, samples] in [-1, 1] at 16 kHz. The audio is encoded into
clean, frozen audio latents that the video attends to while
denoising, and the original samples are muxed into the output
(audio decoding is skipped).}

\item{verbose}{Logical, or one of "silent", "progress", "steps".
TRUE = "steps" (full per-phase chatter, per-step sigma/timing
lines), FALSE = "silent". "progress" prints a one-line generation
summary plus a denoise progress bar (interactive) or periodic
step ticks (captured logs).}

\item{text_encoder,tokenizer}{Gemma3 model and tokenizer (or paths;
see \code{\link{load_gemma3_text_encoder}} and
\code{\link{gemma3_tokenizer}}). Ignored when \code{prompt_embeds}
is supplied.}

\item{width,height}{Integers. Output resolution (multiples of 32).}

\item{decode_video,decode_audio}{Logicals. Decode the respective
latents (disable for latent-space work).}
}
\value{
Invisibly, a list with \code{video} (array
  [frames, height, width, 3] in [0, 1]), \code{audio} (matrix
  [2, samples] in [-1, 1]), \code{sample_rate}, the raw
  \code{latents} and \code{audio_latents}, and \code{latent_shape}
  (c(frames, height, width) of the latent geometry, for
  \code{\link{ltx23_tail_latents}}).
}
\description{
Distilled text-to-video generation: encodes the prompt with Gemma3 +
connectors, denoises joint audio/video latents over the official
8-step distilled schedule (no classifier-free guidance), decodes the
video with the causal VAE and the audio with the audio VAE + BWE
vocoder, and optionally muxes both into an MP4.
}
