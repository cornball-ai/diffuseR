% tinyrox says don't edit this manually, but it can't stop you!
\name{txt2vid_animatediff}
\alias{txt2vid_animatediff}
\title{Generate a video with AnimateDiff (SD 1.5)}
\usage{
txt2vid_animatediff(
  prompt,
  negative_prompt = NULL,
  pipeline = NULL,
  width = 512L,
  height = 512L,
  num_frames = 16L,
  context_length = 16L,
  context_stride = 4L,
  num_inference_steps = 25L,
  guidance_scale = 7.5,
  seed = NULL,
  latents = NULL,
  decode_chunk = 1L,
  fps = 8L,
  filename = NULL,
  save_file = TRUE,
  verbose = TRUE
)
}
\arguments{
\item{prompt}{A prompt, or keyframes: a character vector or list named
by 1-based frame number, e.g. \code{c("1" = "a red fox", "16" = "a
wolf made of smoke")}. Frames between keyframes blend the two prompt
embeddings.}

\item{negative_prompt}{Negative prompt (one for the whole clip).}

\item{pipeline}{From \code{\link{animatediff_load_pipeline}}; NULL loads
the default (v3, CUDA).}

\item{width}{Output width, a multiple of 8 (SD 1.5 is trained at 512).}

\item{height}{Output height, a multiple of 8.}

\item{num_frames}{Frames. Past \code{context_length} the motion
modules attend within overlapping windows (FreeNoise), so clips can
be longer than the 16 frames the modules were trained on.}

\item{context_length}{Frames per motion window, at most the
pipeline's \code{train_frames} (16): the v2 and v3 modules turn to
noise over longer spans even though their position table covers 32.}

\item{context_stride}{Frames between window starts.}

\item{num_inference_steps}{DDIM steps.}

\item{guidance_scale}{Classifier-free guidance. Higher values (10-15)
push harder towards the prompt and can add artifacts.}

\item{seed}{Integer seed, or NULL.}

\item{latents}{Optional starting noise \code{[1, 4, F, H/8, W/8]}.}

\item{decode_chunk}{Frames per VAE decode. The fp32 decoder needs about
2 GiB per 512x512 frame, so 4 at once set a 13.9 GiB peak where 1
costs 5 GiB and about 2 s.}

\item{fps}{Frame rate of the saved file.}

\item{filename}{Output video path; NULL with \code{save_file = TRUE}
derives one from the prompt.}

\item{save_file}{Logical; write the video with \code{\link{save_video}}.}

\item{verbose}{Logical.}
}
\value{
A list: \code{video} (array \code{[F, H, W, 3]} in 0..1),
  \code{file} (or NULL) and \code{metadata}.
}
\description{
Text-to-video from pure noise. The image model draws every frame and the
motion module ties them together loosely, so objects drift and morph;
\code{prompt} keyframes push that further by changing the subject over
the clip.
}
\examples{
\dontrun{
pipe <- animatediff_load_pipeline("v3")
txt2vid_animatediff("a lighthouse in a storm, oil painting",
                    pipeline = pipe, seed = 7, filename = "storm.mp4")
# prompt travel: the subject melts from one prompt into the next
txt2vid_animatediff(c("1" = "a red fox in snow", "16" = "a wolf made of smoke"),
                    pipeline = pipe, guidance_scale = 12, filename = "fox.mp4")
}
}
