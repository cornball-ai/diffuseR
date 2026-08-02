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
components based on available VRAM (via nvidia-smi) and GPU architecture.
}
\details{
Strategies:
\describe{
  \item{"auto"}{Detect free VRAM and choose the best strategy}
  \item{"full_gpu"}{All components on CUDA (10GB+ free VRAM for SDXL)}
  \item{"unet_gpu"}{Only unet on CUDA, rest on CPU (6GB+ for SDXL)}
  \item{"cpu_only"}{All components on CPU}
}

On Blackwell GPUs (RTX 50xx), "unet_gpu" is forced due to TorchScript
compatibility issues, regardless of available VRAM. The native modules
(`use_native_unet` and friends) do not have this restriction.
}
\examples{
# Force a strategy: no GPU or nvidia-smi needed.
str(auto_devices("sdxl", strategy = "cpu_only"))

str(auto_devices("sd21", strategy = "unet_gpu"))

# Auto-detect free VRAM and pick a strategy for this machine.
str(auto_devices("sdxl"))
}
