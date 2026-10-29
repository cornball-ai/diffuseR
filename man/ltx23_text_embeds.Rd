% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_text_embeds}
\alias{ltx23_text_embeds}
\title{Encode an LTX prompt to connector embeddings}
\usage{
ltx23_text_embeds(
  prompt,
  pipeline,
  text_encoder = NULL,
  tokenizer = NULL,
  prompt_embeds = NULL,
  device = "cuda",
  dtype = "bfloat16",
  max_sequence_length = 1024L,
  verbose = TRUE
)
}
\arguments{
\item{prompt}{Character. The prompt.}

\item{pipeline}{An \code{ltx23_pipeline} holding the connectors, e.g.
\code{ltx23_load_pipeline(path, components = "connectors")}.}

\item{prompt_embeds}{Optional raw encoder output (the result of
\code{\link{encode_with_gemma4}} or \code{\link{encode_with_gemma3}}),
so the encoder can be freed before the connectors load.}

\item{device}{Character. Compute device.}

\item{dtype}{Character. Compute dtype.}

\item{max_sequence_length}{Integer. Text token length.}

\item{verbose}{Logical.}

\item{text_encoder,tokenizer}{The Gemma 3 or Gemma 4 model and
tokenizer (see \code{\link{txt2vid_ltx2}}). Ignored when
\code{prompt_embeds} is supplied.}
}
\value{
A list with \code{video_text_embedding},
  \code{audio_text_embedding} and \code{attention_mask}, on the CPU.
}
\description{
Runs the text encoder and the text connectors and returns their
output on the CPU, ready for the \code{connector_embeds} argument of
\code{\link{txt2vid_ltx2}}. This splits a render into phases that
never share host RAM: encode with Gemma and drop it, run a
connectors-only pipeline (loaded with \code{pin = FALSE}) and drop
it, then load the render pipeline without connectors.
}
\examples{
\dontrun{
ckpt <- file.path(tools::R_user_dir("diffuseR", "data"), "ltx2.5-nf4-full")
gemma <- file.path(tools::R_user_dir("diffuseR", "data"), "gemma4-nf4")
prompt <- "A red fox in fresh snow"
te <- load_gemma4_nf4(gemma, device = "cpu", stream = TRUE, pin = FALSE)
raw <- encode_with_gemma4(prompt, te, gemma4_tokenizer(gemma))
rm(te); gc()
text <- ltx23_load_pipeline(ckpt, components = "connectors",
                            resident_gb = 4, pin = FALSE)
emb <- ltx23_text_embeds(prompt, text, prompt_embeds = raw)
rm(text, raw); gc()
pipe <- ltx23_load_pipeline(ckpt, components = c("dit", "vae",
                            "audio_vae", "vocoder"), resident_gb = 4,
                            resident_from = "disk")
txt2vid_ltx2(prompt, pipe, connector_embeds = emb, filename = "fox.mp4")
}

}
