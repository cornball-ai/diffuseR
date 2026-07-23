% tinyrox says don't edit this manually, but it can't stop you!
\name{serve}
\alias{serve}
\title{Serve diffuseR over HTTP}
\usage{
serve(port = 7812L, model = c("flux2", "zimage", "flux1", "ltx"),
      device = "cuda", token = NULL, max_pixels = 1024L^2, max_frames = 161L,
      max_steps = 50L, max_pixel_frames = NULL, max_prompts = 32L,
      timeout = 300L, max_body = 1024L^2, warmup = TRUE)
}
\arguments{
\item{port}{Integer. TCP port. Default 7812 (cornball serve range: whisper 7809, chatterbox 7810, qwen3 TTS 7811).}

\item{model}{One of "flux2", "zimage", "flux1" (images) or "ltx"
(video). SD 2.1/SDXL are not served yet.}

\item{device}{Character. "cuda" or "cpu".}

\item{token}{Character or NULL. Shared secret; when set, requests
must send \code{Authorization: Bearer <token>}.}

\item{max_pixels}{Integer. Maximum width x height accepted (images
and video frames). Default 1024^2.}

\item{max_frames}{Integer. Maximum video frame count. Default 161.}

\item{max_steps}{Integer. Maximum image inference steps. Default 50.}

\item{max_pixel_frames}{Numeric. Joint video budget: width x height
x frames must stay under it (NULL = max_pixels x 121, so full-
resolution clips top out at 121 frames and longer clips must
shrink spatially).}

\item{max_prompts}{Integer. Bound on the per-prompt connector-embed
cache for "ltx" (~9 MB per entry, LRU-evicted). Default 32.}

\item{timeout}{Integer. Per-connection I/O timeout in seconds.}

\item{max_body}{Integer. Maximum request body bytes. Default 1 MB
(bodies are JSON).}

\item{warmup}{Logical. Image models: run one small generation at
startup so the first request doesn't pay tracing and allocator
growth. Ignored for "ltx".}
}
\value{
Does not return normally; runs until interrupted.
}
\description{
Starts a blocking HTTP server that loads one model and answers
OpenAI-style generation requests. Never downloads weights: if the
model's artifacts are missing, startup stops with the loader's
pointer to the explicit \code{download_*()} function.
}
\details{
Endpoints:
\itemize{
\item \code{GET /health} - liveness probe, returns
\code{{"status":"ok","model":...}}. The server is
single-threaded, so health only answers between requests.
\item \code{POST /v1/images/generations} - image models. JSON body
\code{{prompt, size, seed, steps}} (\code{size} like
\code{"1024x1024"}; \code{n > 1} is not supported). Returns
\code{{created, data: [{b64_json}]}} with a base64 PNG.
\item \code{POST /v1/videos/generations} - \code{model = "ltx"}
only. JSON body \code{{prompt, width, height, num_frames,
frame_rate, seed}}. Returns raw \code{video/mp4} bytes. Video
generation takes minutes: give your client a matching timeout.
}

The server is single-threaded and runs until interrupted. Run it
under a process supervisor (systemd, tmux); an example unit ships
with the package: \code{system.file("diffuser.service",
package = "diffuseR")}.

Security: base R's \code{serverSocket} binds all interfaces, so the
server is reachable by anything that can reach the machine. Keep it
behind a firewall or reverse proxy, and/or set \code{token}: when
set, every request must carry \code{Authorization: Bearer <token>}
or it is refused with 401. Generation size is capped by
\code{max_pixels}/\code{max_frames}; oversized requests get 400. A
CUDA out-of-memory during a request answers 500 and then exits the
process (status 70) so a supervisor restarts it with clean GPU
state rather than serving on with stranded components.

}
