% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_zimage_generate}
\alias{yq_zimage_generate}
\title{Generate a Z-Image-Turbo image on the anvl backend}
\usage{
yq_zimage_generate(input_ids, attention_mask, noise, height, width, dit_dir,
                   qwen_dir, vae_path, steps = 8L, shift = 3,
                   penult_layer = 35L, decode = TRUE, device = "cpu",
                   precision = "highest", verbose = TRUE)
}
\arguments{
\item{input_ids}{Integer matrix \code{[1, S]} of 0-based Qwen3 token
ids (chat-templated, right-padded), as produced by
\code{encode_qwen(..., enable_thinking = TRUE)}.}

\item{attention_mask}{Integer matrix \code{[1, S]} (1 real, 0 pad).}

\item{noise}{AnvlArray \code{[1, 16, H/8, W/8]} initial latents.}

\item{dit_dir}{The fp8 transformer artifact directory.}

\item{qwen_dir}{The Qwen3 \code{text_encoder} directory (index +
shards).}

\item{vae_path}{Native \code{vae_decoder_native} decoder state_dict
\code{.safetensors} (no \code{decoder.} prefix).}

\item{steps}{Integer. Denoising steps (Turbo: 8).}

\item{shift}{Numeric. FlowMatch static shift (3.0).}

\item{penult_layer}{Integer. Qwen3 layers to run; the penultimate
hidden state (num_hidden_layers - 1 = 35) is the caption feature.}

\item{decode}{Logical. Run the VAE decode (else pixels are NULL).}

\item{device}{Character. Compute device.}

\item{precision}{Character. Matmul precision (\code{"highest"} = strict
f32 for parity).}

\item{verbose}{Logical.}

\item{height,width}{Integers, divisible by 16.}
}
\value{
List \code{list(latents, pixels, cap_feats)}: final latents
  \code{[1, 16, H/8, W/8]}, decoded pixels \code{[1, 3, H, W]} in
  [-1, 1] (or NULL), and the Qwen3 caption features
  \code{[1, n_real, 2560]} (for debugging the text stage).
}
\description{
Runs the full text-to-image pipeline from real weights, one phase
resident at a time (Qwen3 encode, then DiT denoise, then VAE decode),
freeing each phase's weights before the next so the f32 residents do
not accumulate. Deterministic given the token ids and the initial
noise, so it can be validated against a torch reference by feeding the
same fixture inputs.
}
