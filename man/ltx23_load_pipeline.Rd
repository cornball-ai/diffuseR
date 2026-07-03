% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_load_pipeline}
\alias{ltx23_load_pipeline}
\title{Load the LTX-2.3 generation components from a single-file checkpoint}
\usage{
ltx23_load_pipeline(checkpoint_path, device = "cuda", dtype = "bfloat16",
                    transformer_device = "cpu",
                    components = c("dit", "connectors", "vae", "audio_vae", "vocoder"),
                    verbose = TRUE)
}
\arguments{
\item{checkpoint_path}{Path to e.g.
\code{ltx-2.3-22b-distilled-1.1.safetensors}.}

\item{device}{Character. Device for the small components (VAEs,
vocoder, connectors). The transformer placement is controlled by
\code{transformer_device}.}

\item{dtype}{Character. "bfloat16" (checkpoint native) or "float32".}

\item{transformer_device}{Character. Device for the transformer
weights (default "cpu"; the fp8/offload path manages movement).}

\item{components}{Character vector. Which components to load.}

\item{verbose}{Logical.}
}
\value{
A list with the loaded modules and the checkpoint config,
  class \code{ltx23_pipeline}.
}
\description{
Builds the transformer, connectors, video VAE, audio VAE, and vocoder
with the LTX 2.3 configuration and streams the checkpoint weights into
them. The Gemma3 text encoder ships separately (see
\code{\link{load_gemma3_text_encoder}}).
}
