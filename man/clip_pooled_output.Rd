% tinyrox says don't edit this manually, but it can't stop you!
\name{clip_pooled_output}
\alias{clip_pooled_output}
\title{Pooled CLIP output at the EOS position}
\usage{
clip_pooled_output(hidden_states, input_ids)
}
\arguments{
\item{hidden_states}{Final-LN hidden states [B, S, D] from
\code{\link{text_encoder_native}} (with \code{apply_final_ln = TRUE}).}

\item{input_ids}{Token ids [B, S] (0-based, as fed to the encoder).}
}
\value{
Tensor [B, D].
}
\description{
The HF CLIPTextModel pooler_output: the final-layer-norm hidden state
at the EOS token position, located by argmax over the token ids (EOS
is the highest id in the CLIP vocab, and causal attention makes any
padding after it irrelevant). No text projection is applied - this is
what FLUX uses as pooled_projections.
}
