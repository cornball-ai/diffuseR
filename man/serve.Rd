% tinyrox says don't edit this manually, but it can't stop you!
\name{serve}
\alias{serve}
\title{Serve diffuseR over HTTP}
\usage{
serve(port = 7812L, model = c("flux2", "zimage", "flux1", "ltx"),
      device = "cuda", timeout = 300L, max_body = 1024L^2, warmup = TRUE)
}
\arguments{
\item{port}{Integer. TCP port. Default 7812 (cornball serve range: whisper 7809, chatterbox 7810, qwen3 TTS 7811).}

\item{model}{One of "flux2", "zimage", "flux1" (images) or "ltx"
(video). SD 2.1/SDXL are not served yet.}

\item{device}{Character. "cuda" or "cpu".}

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

}
