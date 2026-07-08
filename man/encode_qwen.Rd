% tinyrox says don't edit this manually, but it can't stop you!
\name{encode_qwen}
\alias{encode_qwen}
\title{Encode prompts with the Qwen tokenizer}
\usage{
encode_qwen(tokenizer, texts, max_length = 512L, chat_template = TRUE,
            enable_thinking = FALSE)
}
\arguments{
\item{tokenizer}{A \code{\link{qwen_bpe_tokenizer}}.}

\item{texts}{Character vector of prompts.}

\item{max_length}{Integer. Fixed sequence length (klein: 512). NULL
for no truncation/padding.}

\item{chat_template}{Logical. Wrap in the Qwen3 chat template.}

\item{enable_thinking}{Logical. Leave the model's thinking enabled
(no empty think block). Default FALSE.}
}
\value{
List with \code{input_ids} and \code{attention_mask} integer
  matrices [length(texts), max_length] (ragged lists when
  \code{max_length} is NULL). Ids are 0-based.
}
\description{
With \code{chat_template = TRUE} each prompt is wrapped as a single
user turn with the generation prompt, matching
\code{apply_chat_template(..., add_generation_prompt = TRUE)}. With
\code{enable_thinking = FALSE} (the FLUX.2 klein pipeline behavior)
the template closes with an empty thinking block; with
\code{enable_thinking = TRUE} (the Z-Image pipeline behavior) it ends
at the assistant turn. Right-pads with \code{<|endoftext|>}.
}
