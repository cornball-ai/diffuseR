# diffuseR

[![CRAN status](https://www.r-pkg.org/badges/version/diffuseR)](https://CRAN.R-project.org/package=diffuseR)
[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)

## Overview

`diffuseR` is a functional R implementation of diffusion models, inspired by Hugging Face's Python `diffusers` library. The package provides a simple, idiomatic R interface to state-of-the-art generative AI models for image generation and manipulation using base R and the `torch` package. No Python dependencies.

```r
# Simple text-to-image generation
library(diffuseR)
image <- txt2img("A serene landscape with mountains and a lake at sunset", save_to = "landscape.png")
```

## Installation

First install torch. As per [this comment](https://github.com/mlverse/torch/issues/1198#issuecomment-2419363312), using the pre-built binaries from [https://torch.mlverse.org/docs/articles/installation#pre-built](https://torch.mlverse.org/docs/articles/installation#pre-built) are heavily recommend. The pre-built binaries bundle the necessary CUDA and cudnn versions, so you don't need a global compatible system version of CUDA:

```r
options(timeout = 600) # increasing timeout is recommended since we will be downloading a 2GB file.
# For Windows and Linux: "cpu", "cu124" are the only currently supported
# For MacOS the supported are: "cpu-intel" or "cpu-m1"
kind <- "cu124"
version <- available.packages()["torch","Version"]
options(repos = c(
  torch = sprintf("https://torch-cdn.mlverse.org/packages/%s/%s/", kind, version),
  CRAN = "https://cloud.r-project.org" # or any other from which you want to install the other R dependencies.
))
install.packages("torch")
```

You can install the development version of diffuseR from GitHub:

```r
# install.packages("devtools")
devtools::install_github("yourusername/diffuseR")
# Or
# install.packages("targets")
targets::install_github("yourusername/diffuseR")
```

## Features

- **Text-to-Image Generation**: Create images from textual descriptions
- **Image-to-Image Generation**: Modify existing images based on text prompts
- **Multiple Models**: Support for Stable Diffusion 2.1 and SDXL (more coming soon)
- **Scheduler Options**: DDIM (more coming soon)
- **Device Support**: Works on both CPU and GPU
- **R-native Interface**: Functional programming approach that feels natural in R

## Quick Start

### Basic Usage

```r
library(diffuseR)
torch::local_no_grad()

# Download the Stable Diffusion 2.1 models
sd <- load_pipeline("sd21")

# Generate an image from text
cat_img <- txt2img(
  prompt = "a photorealistic cat wearing sunglasses",
  steps = 30,
  save_to = "cat.png"
)
```

### Advanced Usage with GPU

```r
# Load and configure a pipeline once
sd <- load_pipeline("stable-diffusion-2-1", device = "cuda")

# Change the scheduler
sd$scheduler <- create_scheduler("dpm_solver_plus", num_inference_steps = 20)

# Generate multiple images with the same configuration
mountain_img <- generate(sd, prompt = "mountains at sunset")
ocean_img <- generate(sd, prompt = "waves crashing on the shore")

# Save images
save_image(mountain_img, "mountain.png")
save_image(ocean_img, "ocean.png")
```

## Supported Models

Currently supported models:

- Stable Diffusion 2.1
- Stable Diffusion XL (SDXL)
- More coming soon!

## Roadmap

Future plans for diffuseR include:

- [ ] Pipeline mode to load models once and reuse
- [ ] Inpainting support
- [ ] Additional schedulers (PNDM, DPMSolverMultistep, Euler ancestral)
- [ ] text-to-video generation

## How It Works

diffuseR uses TorchScript models exported from PyTorch implementations for the deep learning parts of the implementation. This approach was the quickest and easiest way to build the machinery that supports diffusion models in R. Full R torch implementations of the models are planned for the future, but this initial version allows users to quickly get started with diffusion models in R without needing to rely on Python.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache 2. License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for the original diffusers library
- Stability AI for Stable Diffusion
- The R and torch communities for their excellent tooling
