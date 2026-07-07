% tinyrox says don't edit this manually, but it can't stop you!
\name{encode_qwen}
\alias{encode_qwen}
\title{Encode prompts with the Qwen tokenizer}
\usage{
encode_qwen(tokenizer, texts, max_length = 512L, chat_template = TRUE)
}
\arguments{
\item{tokenizer}{A \code{\link{qwen_bpe_tokenizer}}.}

\item{texts}{Character vector of prompts.}

\item{max_length}{Integer. Fixed sequence length (klein: 512). NULL
for no truncation/padding.}

\item{chat_template}{Logical. Wrap in the Qwen3 chat template.}
}
\value{
List with \code{input_ids} and \code{attention_mask} integer
  matrices [length(texts), max_length] (ragged lists when
  \code{max_length} is NULL). Ids are 0-based.
}
\description{
With \code{chat_template = TRUE} (the FLUX.2 klein pipeline behavior)
each prompt is wrapped as a single user turn with the generation
prompt and a disabled thinking block, matching
\code{apply_chat_template(..., add_generation_prompt = TRUE,
enable_thinking = FALSE)}. Right-pads with \code{<|endoftext|>}.
}
