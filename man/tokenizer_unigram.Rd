% tinyrox says don't edit this manually, but it can't stop you!
\name{tokenizer_unigram}
\alias{tokenizer_unigram}
\title{SentencePiece Unigram Tokenizer}
\description{
Pure R implementation of HuggingFace tokenizer.json files with a
Unigram model (SentencePiece), as used by T5 - FLUX's second text
encoder. Segmentation is Viterbi best-path over the vocab log
probabilities (Kudo 2018, arXiv:1804.10959). The normalizer and
Metaspace pre-tokenizer settings are read from the file.
}
\details{
Limitation: the Precompiled charsmap normalizer (NFKC-style unicode
mapping) is approximated by control-whitespace substitution only;
ASCII and common latin text tokenizes identically to the reference,
exotic unicode may differ.

}
