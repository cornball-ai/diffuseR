% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_ltx_vae_load_weights}
\alias{yq_ltx_vae_load_weights}
\title{Load LTX-2.3 VAE decoder weights into an anvl pytree}
\usage{
yq_ltx_vae_load_weights(path, device = "cpu", strict = TRUE, patch_size = 4L,
                        patch_size_t = 1L)
}
\arguments{
\item{path}{Path to the decoder state_dict \code{.safetensors} (native
\code{ltx23_video_decoder3d} key names, no \code{decoder.} prefix).}

\item{device}{Character. Target device.}

\item{strict}{Logical. Enforce the exact key census.}

\item{patch_size,patch_size_t}{Integers. Pixel un-patchification factors
(LTX 2.3 defaults 4 and 1).}
}
\value{
Weights pytree for \code{\link{yq_ltx_vae_decode}}.
}
\description{
Reads a \code{diffuseR::ltx23_video_decoder3d} state_dict (F16/F32
upcast to f32): \code{conv_in}, the mid ResNet stack, the four up
blocks (each an optional channel-changing ResNet \code{conv_in}, an
optional \code{upsamplers.0} pixel-shuffle conv, and a ResNet stack),
and \code{conv_out}. The per-channel RMS norms carry no weights, so the
entire census is Conv3d \code{weight}/\code{bias} pairs. Conv weights
load raw \code{[out, in, k_t, k_h, k_w]} (what
\code{\link[anvl]{prim_convolution}} expects). Resnet counts and up
block count are probed by key existence; the fixed LTX-2.3 up-block
pixel-shuffle geometry (strides \code{(2,2,2)}, \code{(2,2,2)},
\code{(2,1,1)}, \code{(1,2,2)}; upscale factors \code{2, 1, 2, 2}) is
hard-coded. With \code{strict = TRUE} every file key must be consumed
exactly once.
}
