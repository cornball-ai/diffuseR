% tinyrox says don't edit this manually, but it can't stop you!
\name{flux2_double_block}
\alias{flux2_double_block}
\title{FLUX.2 double-stream (MMDiT) block}
\usage{
flux2_double_block(dim, num_attention_heads, attention_head_dim, mlp_ratio = 3,
                   eps = 1e-06, bias = FALSE)
}
\arguments{
\item{dim}{Integer. Model dimension.}

\item{num_attention_heads}{Integer. Attention heads.}

\item{attention_head_dim}{Integer. Per-head dimension.}

\item{mlp_ratio}{Numeric. FF multiplier (FLUX.2: 3.0).}

\item{eps}{Numeric. Norm epsilon.}

\item{bias}{Logical.}
}
\value{
Module whose forward(hidden_states, encoder_hidden_states,
  temb_mod_img, temb_mod_txt, image_rotary_emb) returns
  \code{list(encoder_hidden_states, hidden_states)}.
}
\description{
Image and text streams with externally supplied (shift, scale, gate)
modulation triples, joint attention (txt first), and SwiGLU
feed-forwards. Reference: Flux2TransformerBlock.
}
