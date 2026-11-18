% tinyrox says don't edit this manually, but it can't stop you!
\name{audio2vid_ltx2}
\alias{audio2vid_ltx2}
\title{Long audio-driven video with LTX-2}
\usage{
audio2vid_ltx2(
  audio,
  image,
  prompt,
  pipeline = NULL,
  output_dir,
  width = 768L,
  height = 512L,
  window = 97L,
  carry = 25L,
  windows = NULL,
  frame_rate = 24,
  seed = NULL,
  resume = TRUE,
  on_window = NULL,
  verbose = "progress",
  generate = NULL,
  ...
)
}
\arguments{
\item{audio}{The narration: a file path (read with
\code{\link{ltx23_read_audio}}, which needs \code{av}) or a matrix
[2, samples] in [-1, 1] at 16 kHz.}

\item{image}{Start frame of the first window: a PNG/JPEG path or an
[H, W, 3] array in [0, 1].}

\item{prompt}{Character. One prompt for every window, or one per
window.}

\item{pipeline}{An \code{ltx23_pipeline} (from
\code{\link{ltx23_load_pipeline}}) or a resident LTX handle (from
\code{\link{resident_load}}, active).}

\item{output_dir}{Directory for the windows; created if needed.}

\item{windows}{Optional plan: a data frame with \code{start} and
\code{frames} (8k+1), each window starting \code{carry} frames before
the previous one ends, the first at 0. Default
\code{ltx2_windows(<narration frames>, window, carry)}.}

\item{frame_rate}{Frames per second.}

\item{seed}{Optional integer; window j renders with \code{seed + j - 1}.}

\item{resume}{Logical. Keep windows already on disk that match the plan.}

\item{on_window}{Optional function called with each window's record (a
list: \code{index}, \code{start}, \code{frames}, \code{file},
\code{latents}, \code{rendered}, \code{seconds}) once it is on disk.
An error raised there stops the run after that window; the windows
written so far stay, so a later call resumes.}

\item{verbose}{Passed to \code{\link{txt2vid_ltx2}}.}

\item{generate}{Optional generator \code{function(prompt, ...)} in place
of the one \code{pipeline} implies; for tests and hosts.}

\item{...}{Further arguments for \code{\link{txt2vid_ltx2}}
(\code{text_encoder}, \code{tokenizer}, \code{resident}, ...).}

\item{width,height}{Frame size, multiples of 32.}

\item{window,carry}{Window length and overlap in frames, 8k+1 (see
\code{\link{ltx2_windows}}). Ignored when \code{windows} is given,
except \code{carry}, which every plan shares.}
}
\value{
Invisibly, a data frame with one row per window:
  \code{index}, \code{start}, \code{frames}, \code{file},
  \code{latents}, \code{prompt} and \code{rendered} (FALSE when kept
  from an earlier run).
}
\description{
Renders a narration of any length as a chain of LTX windows. The first
window starts from \code{image}; each later window is conditioned on
the previous window's last \code{carry} frames as latents, and every
window is driven by the narration over its own span. Each window is
written to \code{output_dir} as it finishes: \code{windowNN.mp4} (the
window's frames, its carried head included, with its audio),
\code{windowNN.latents.pt} (its tail latents) and \code{windowNN.json}
(its plan and prompt). Joining the windows (crossfading each carried
head over the previous window's tail) is left to the caller.
}
\details{
A second call with the same arguments resumes: a window whose three
files exist and whose record matches the plan is kept, and the chain
continues from its saved latents. Once one window is rendered again,
every later window is too, since its conditioning changed.

}
