% tinyrox says don't edit this manually, but it can't stop you!
\name{gemma3_encode_batch}
\alias{gemma3_encode_batch}
\title{Batch-encode prompts with Gemma3, cached to disk}
\usage{
gemma3_encode_batch(prompts, model = NULL, tokenizer = NULL, batch_size = 4L,
                    cache_dir = NULL, max_sequence_length = 1024L,
                    device = "cuda", verbose = TRUE)
}
\arguments{
\item{prompts}{Character vector.}

\item{batch_size}{Integer. Prompts per forward pass (default 4;
raise on cards with headroom, lower if the encode OOMs).}

\item{cache_dir}{Optional directory. When given, each prompt is
written to \code{gemma3-<md5>.pt} and the return value is the
character vector of those paths (load one with
\code{torch::torch_load}); when NULL, the return value is a list
of \code{list(prompt_embeds, prompt_attention_mask)} in prompt
order.}

\item{model,tokenizer}{As in \code{\link{encode_with_gemma3}}.}

\item{max_sequence_length,device,verbose}{As in
\code{\link{encode_with_gemma3}}.}
}
\value{
Character vector of cache paths (with \code{cache_dir}) or
  a list of per-prompt embedding results (without).
}
\description{
Encodes a vector of prompts in sub-batches (bounding activation
VRAM) and optionally caches each prompt's result under
\code{cache_dir}, keyed by the prompt text and sequence length.
Already-cached prompts are skipped, so an interrupted batch resumes
where it stopped. Embeddings land on the CPU either way; the
renderer moves them per phase.
}
\details{
Budget note: each prompt's result is the full hidden-state stack the
LTX connectors consume - roughly 0.4 GB at 1024 tokens - so caching
N prompts needs ~0.4N GB of disk.

}
