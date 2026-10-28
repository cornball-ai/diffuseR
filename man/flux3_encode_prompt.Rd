% tinyrox says don't edit this manually, but it can't stop you!
\name{flux3_encode_prompt}
\alias{flux3_encode_prompt}
\title{Encode a prompt for FLUX 3}
\usage{
flux3_encode_prompt(
  prompt,
  model,
  tokenizer,
  max_sequence_length = 8192L,
  pad_multiple = 80L,
  out_layers = seq(4L, 32L, by = 4L),
  device = NULL
)
}
\arguments{
\item{prompt}{Character scalar.}

\item{model}{A \code{\link{qwen3_encoder}} loaded from FLUX 3's
\code{text_encoder} with \code{\link{load_qwen3_text_encoder}}.}

\item{tokenizer}{A \code{\link{qwen_bpe_tokenizer}} from the same
directory.}

\item{max_sequence_length}{Integer. Token cap (8192).}

\item{pad_multiple}{Integer. Window rounding (80).}

\item{out_layers}{Integer vector. Hidden-state layers to stack
(4, 8, ..., 32).}

\item{device}{Character or NULL. Defaults to the model's device.}
}
\value{
\code{list(prompt_embeds, txt_ids)}: embeddings [1, L,
  length(out_layers) * hidden] and position ids [L, 4] from
  \code{\link{flux3_prepare_text_ids}}.
}
\description{
The Qwen3-VL prompt path of the reference pipeline (diffusers
Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds, Apache-2.0): the
prompt goes through the chat template (one user turn plus the
generation prompt, no thinking block), is right-padded to a multiple
of \code{pad_multiple} tokens (capped at \code{max_sequence_length}),
and the hidden states of \code{out_layers} are concatenated per
token. The padding tokens get position id 1, as in the training
environment, and the transformer attends to them, so their states are
part of the embedding.
}
\details{
FLUX 3 expects a full scene caption: a short prompt gives generic
output. Expand short requests with an LLM first (BFL publishes the
system prompts with flux-3-image-klein-pup).

}
