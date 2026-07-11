% tinyrox says don't edit this manually, but it can't stop you!
\name{text_encoder_native}
\alias{text_encoder_native}
\title{Native CLIP Text Encoder}
\usage{
text_encoder_native(vocab_size = 49408, context_length = 77, embed_dim = 768,
                    num_layers = 12, num_heads = 12, mlp_dim = 3072,
                    apply_final_ln = TRUE, return_penultimate = FALSE,
                    gelu_type = "tanh")
}
\arguments{
\item{vocab_size}{Vocabulary size (default 49408)}

\item{context_length}{Maximum sequence length (default 77)}

\item{embed_dim}{Embedding dimension}

\item{num_layers}{Number of transformer layers}

\item{num_heads}{Number of attention heads}

\item{mlp_dim}{MLP hidden dimension}

\item{apply_final_ln}{Whether to apply final layer norm (default TRUE).
Set to FALSE to match TorchScript exports that don't include final LN.}

\item{return_penultimate}{Return the second-to-last transformer block's
output (\code{hidden_states[-2]}, pre-final-LN) instead of the last
layer. This is what SDXL feeds the UNet cross-attention; final LN is
never applied to the penultimate output (default FALSE).}

\item{gelu_type}{GELU variant: "tanh" (matches the TorchScript exports),
"quick" (HF CLIP ViT-L, used by SDXL text_encoder and FLUX), or "exact"}
}
\value{
An nn_module representing the text encoder
}
\description{
Native R torch implementation of CLIP text encoder.
Replaces TorchScript for better GPU compatibility.
}
