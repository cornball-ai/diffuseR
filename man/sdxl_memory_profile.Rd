% tinyrox says don't edit this manually, but it can't stop you!
\name{sdxl_memory_profile}
\alias{sdxl_memory_profile}
\title{Get SDXL Memory Profile}
\usage{
sdxl_memory_profile(vram_gb = NULL)
}
\arguments{
\item{vram_gb}{Numeric. Available VRAM in GB, or NULL for auto-detection.}
}
\value{
A list with memory profile settings.
}
\description{
Determines optimal memory configuration for SDXL image generation
based on available VRAM.
}
\details{
Memory profiles for SDXL:
\describe{
  \item{full_gpu}{16GB+ - All components on CUDA}
  \item{balanced}{10-12GB - UNet + decoder on CUDA, text encoders on CPU}
  \item{unet_gpu}{6-10GB - Only UNet on CUDA, everything else CPU}
  \item{cpu_only}{<6GB - All on CPU}
}

Each profile also specifies:
- cfg_mode: "batched" or "sequential" (sequential halves peak memory)
- cleanup: "none", "phase", or "step" (when to clear VRAM)
- dtype: "float16" or "float32"
- max_resolution: maximum image dimension
}
\examples{
\dontrun{
# Auto-detect profile
profile <- sdxl_memory_profile()

# Specific VRAM
profile <- sdxl_memory_profile(vram_gb = 8)
}
}
