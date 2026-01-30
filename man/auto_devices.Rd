% tinyrox says don't edit this manually, but it can't stop you!
\name{auto_devices}
\alias{auto_devices}
\title{Auto-Configure Device Assignment}
\usage{
auto_devices(model = "sdxl", strategy = "auto")
}
\arguments{
\item{model}{Character. Model type: "sd21" or "sdxl".}

\item{strategy}{Character. Memory strategy: "auto" (default), "full_gpu",
"unet_gpu", or "cpu_only". See Details.}
}
\value{
A named list of device assignments suitable for `models2devices()`.
}
\description{
Automatically determines optimal device configuration for diffusion model
components based on available VRAM and GPU architecture. Uses gpuctl for
detection if available, otherwise falls back to sensible defaults.
}
\details{
Strategies:
\describe{
  \item{"auto"}{Detect VRAM and choose best strategy (requires gpuctl)}
  \item{"full_gpu"}{All components on CUDA (16GB+ VRAM for SDXL)}
  \item{"unet_gpu"}{Only unet on CUDA, rest on CPU (8GB+ VRAM)}
  \item{"cpu_only"}{All components on CPU}
}

If gpuctl is not installed, "auto" falls back to "unet_gpu" which works on
most modern GPUs (8GB+ VRAM).

On Blackwell GPUs (RTX 50xx), "unet_gpu" is forced due to TorchScript
compatibility issues, regardless of available VRAM.
}
\examples{
\dontrun{
# Auto-detect best configuration
devices <- auto_devices("sdxl")

# Use with models2devices
m2d <- models2devices("sdxl", devices = auto_devices("sdxl"))

# Force CPU-only
devices <- auto_devices("sdxl", strategy = "cpu_only")
}
}
