% tinyrox says don't edit this manually, but it can't stop you!
\name{load_gemma3_text_encoder}
\alias{load_gemma3_text_encoder}
\title{Load Gemma3 Text Model from safetensors}
\usage{
load_gemma3_text_encoder(model_path, device = "cpu", dtype = "float16",
                         pin = getOption("diffuseR.pin_staging", TRUE),
                         verbose = TRUE)
}
\arguments{
\item{model_path}{Character. Path to directory containing model files.}

\item{device}{Character. Device to load model to.}

\item{dtype}{Character. Data type ("float32", "float16", "bfloat16").}

\item{pin}{Logical. When loading to the CPU, page-lock the weights
so \code{\link{encode_with_gemma3}} can swap the model to the GPU
at DMA speed per encode (~0.3 s on, free off) instead of holding
VRAM or reloading. Default follows
\code{options(diffuseR.pin_staging)}.}

\item{verbose}{Logical. Print loading progress.}
}
\value{
Initialized gemma3_text_model with loaded weights.
}
\description{
Loads pre-trained Gemma3 weights from HuggingFace safetensors files.
An NF4 artifact directory (from \code{\link{gemma3_quantize_nf4}})
dispatches to \code{\link{load_gemma3_nf4}}.
}
