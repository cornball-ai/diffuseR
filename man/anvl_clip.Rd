% tinyrox says don't edit this manually, but it can't stop you!
\name{anvl_clip}
\alias{anvl_clip}
\title{Stable Diffusion 2.1 CLIP text encoder (anvl port of text_encoder_native)}
\description{
anvl re-implementation of \code{diffuseR::text_encoder_native} for the
SD 2.1 conditioning path: the OpenCLIP ViT-H text transformer
(\code{embed_dim} 1024, 23 layers, 16 heads, \code{head_dim} 64,
context length 77). Learned token + positional embeddings (host-side
gather, no RoPE), a stack of pre-LayerNorm transformer layers (standard
multi-head self-attention with a causal additive mask, then a
GELU MLP), and a final LayerNorm.
}
\details{
The SD 2.1 pipeline consumes the \strong{final-LayerNorm last hidden
state} (\code{apply_final_ln = TRUE} in
\code{diffuseR::sd_pipeline_from_safetensors}), not a penultimate
\code{clip_skip} layer. The MLP uses the \strong{tanh-approximation
GELU} (\code{diffuseR::text_encoder_native}'s default
\code{gelu_type = "tanh"}, since the minimized SD 2.1 config carries no
\code{hidden_act}), matched here by \code{.yq_clip_gelu}. LayerNorm is
affine with eps 1e-5 (torch \code{nn_layer_norm} default).

}
