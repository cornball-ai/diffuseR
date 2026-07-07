% tinyrox says don't edit this manually, but it can't stop you!
\name{t5_text_encoder}
\alias{t5_text_encoder}
\title{T5 Text Encoder (T5-v1.1)}
\description{
Fresh R port of the T5 encoder stack from HuggingFace transformers
(Apache-2.0, src/transformers/models/t5/modeling_t5.py), as used by
FLUX's second text encoder (T5-v1.1-XXL: 24 layers, d_model 4096,
64 heads x d_kv 64, gated-GELU FFN). Distinctives faithfully carried
over: RMS layer norms (no mean subtraction), no biases anywhere, no
1/sqrt(d) attention scaling (folded into the weights), and a shared
relative position bias computed once from block 1's embedding and
added to every layer's attention logits. Module field names mirror
the checkpoint keys (minus the \code{encoder.} prefix).
}
\details{
FLUX passes no attention mask - padding tokens attend and are
attended to - so none is implemented.

}
