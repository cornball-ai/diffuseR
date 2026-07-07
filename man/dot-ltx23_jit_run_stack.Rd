% tinyrox says don't edit this manually, but it can't stop you!
\name{.ltx23_jit_run_stack}
\alias{.ltx23_jit_run_stack}
\title{Run the block stack through the compiled TorchScript path}
\usage{
.ltx23_jit_run_stack(blocks, hidden_states, audio_hidden_states,
                     encoder_hidden_states, audio_encoder_hidden_states, temb,
                     temb_audio, temb_ca_scale_shift,
                     temb_ca_audio_scale_shift, temb_ca_gate,
                     temb_ca_audio_gate, temb_prompt, temb_prompt_audio,
                     video_rotary_emb, audio_rotary_emb, ca_video_rotary_emb,
                     ca_audio_rotary_emb, encoder_attention_mask = NULL,
                     audio_encoder_attention_mask = NULL,
                     cond_token_index = NULL)
}
\value{
list(hidden_states, audio_hidden_states)
}
\description{
One R-to-libtorch crossing for all blocks: no per-op dispatch, no R
tensor garbage, fused SDPA. Masks must already be additive
\code{[B, 1, 1, S]} (or NULL); rope tensors are the \code{[.., r]}
cos/sin pairs used by the eager path.
}
\keyword{internal}
