% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_zimage_dit}
\alias{yq_zimage_dit}
\title{Z-Image DiT transformer (anvl port)}
\usage{
yq_zimage_dit(heads = 30L, eps = 1e-05, final_eps = 1e-06, precision = "highest")
}
\arguments{
\item{heads}{Integer. Attention heads (full model: 30). Head dim is
derived as \code{dim / heads}.}

\item{eps}{Numeric. RMSNorm epsilon for the blocks (1e-5).}

\item{final_eps}{Numeric. LayerNorm epsilon for the final layer (1e-6).}

\item{precision}{Character. Matmul precision (see
\code{\link[yunque]{yq_linear}}).}
}
\value{
Function of (tokens, cap_feats, t_freq, cos_img, sin_img,
  cos_cap, sin_cap, w):
  \itemize{
    \item tokens \code{[1, img_len, patch_dim]} patchified latent
    \item cap_feats \code{[1, cap_len, cap_feat_dim]} caption features
    \item t_freq \code{[1, 256]} timestep sinusoid
      (\code{\link{yq_zimage_time_embed}})
    \item cos_img, sin_img \code{[img_padded, head_dim]} image RoPE
    \item cos_cap, sin_cap \code{[cap_padded, head_dim]} caption RoPE
    \item w weights pytree (\code{\link{yq_zimage_load_weights}})
  }
  returning the packed final-layer output \code{[1, S_all, patch_dim]}.
}
\description{
anvl re-implementation of \code{diffuseR::zimage_transformer}
(ZImageTransformer2DModel): a single-stream DiT where patchified image
tokens pass through a timestep-modulated noise refiner, caption tokens
through an unmodulated context refiner, then both are concatenated
(image first) and run through the main trunk before an adaLN final
layer projects back to patch space.
}
\details{
Every block uses sandwich RMS norms (a learned norm before AND after
both attention and feed-forward) and a scale/gate-only modulation:
four chunks (scale_msa, gate_msa, scale_mlp, gate_mlp), no shift, gates
\code{tanh}-squashed, scales \code{1 + x}. Attention is plain joint
self-attention with per-head RMS q/k norms and 3-axis interleaved RoPE
(theta 256). The context refiner blocks carry no adaLN weights.

Returns a closure over the static config; \code{anvl::jit()} it. The
timestep sinusoid (\code{\link{yq_zimage_time_embed}}) and RoPE tables
(\code{\link{yq_zimage_rope}}) are precomputed host-side and passed as
inputs; the image patchify happens host-side too (the closure consumes
patchified tokens and returns the packed final-layer output, matching
the FLUX.2 boundary). Batch-of-1.

}
