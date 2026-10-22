% tinyrox says don't edit this manually, but it can't stop you!
\name{tokenizer_gemma4}
\alias{tokenizer_gemma4}
\title{Gemma 4 SentencePiece-style BPE tokenizer}
\description{
Native R port of the HuggingFace \code{tokenizers} pipeline in the
Gemma 4 \code{tokenizer.json} shipped with LTX-2.5: added-token
literals split out first, the \code{Replace " " -> U+2581} normalizer,
characters mapped to vocab ids with \code{<0xHH>} byte fallback, then
rank-ordered BPE merges over the whole string (the \code{Split} on
" " pre-tokenizer never matches after the normalizer). The
post-processor template decides special tokens; for Gemma 4 it adds
none, so there is no BOS.
}
