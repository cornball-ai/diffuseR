% tinyrox says don't edit this manually, but it can't stop you!
\name{anvl_sdxl_clip}
\alias{anvl_sdxl_clip}
\title{Stable Diffusion XL dual CLIP text encoders (anvl port)}
\description{
anvl re-implementation of the two SDXL conditioning encoders,
\code{diffuseR::text_encoder_native} (OpenAI CLIP ViT-L/14 text:
\code{embed_dim} 768, 12 layers, 12 heads, \code{head_dim} 64,
quick-GELU) and \code{diffuseR::text_encoder2_native} (OpenCLIP
ViT-bigG text: \code{embed_dim} 1280, 32 layers, 20 heads,
\code{head_dim} 64, exact/erf GELU). Both share the CLIP
text-transformer structure the SD 2.1 port uses (host-side token +
positional embedding gather, N pre-LayerNorm transformer layers with a
causal additive mask, a GELU MLP), parameterized here per encoder.
}
\details{
\strong{Penultimate selection.} SDXL conditions on the
\emph{penultimate} hidden state of \emph{each} encoder
(\code{hidden_states[-2]} in diffusers: the output of the
second-to-last transformer layer, \strong{before} the final
LayerNorm), not the final-LN state SD 2.1 used. For an N-layer
encoder that is the hidden state after \code{N - 1} layers. The two
penultimate states are concatenated along the feature dim to
\code{[B, seq, 768 + 1280 = 2048]}, the UNet's
\code{cross_attention_dim}.

\strong{Pooled embed.} bigG additionally produces the pooled vector the
UNet's text-time add-embedding consumes: run \emph{all} 32 layers,
apply the final LayerNorm, take the hidden state at the EOS token
position (argmax over the ids), and project through bigG's bias-free
\code{text_projection} to \code{[B, 1280]}.

Per-encoder GELU: CLIP-L uses OpenAI quick-GELU
(\code{x * sigmoid(1.702 x)}); bigG uses the exact erf GELU (matching
\code{text_encoder2_native}'s \code{gelu_type = "exact"}). LayerNorm is
affine with eps 1e-5 (torch \code{nn_layer_norm} default).

}
