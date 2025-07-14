![](man/figures/diffuseRlogo.png)

# diffuseR

[![CRAN status](https://www.r-pkg.org/badges/version/diffuseR)](https://CRAN.R-project.org/package=diffuseR)
[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)

## Overview

`diffuseR` is a functional R implementation of diffusion models, inspired by Hugging Face's Python `diffusers` library. The package provides a simple, idiomatic R interface to state-of-the-art generative AI models for image generation and manipulation using base R and the `torch` package. No Python dependencies. Currently supports Windows and Linux cpu and cuda devices.

## Example output

![](man/figures/20250602_231518_retro_tin_toy_robot.png)
![](man/figures/20250528_200344_Calvin_and_Hobbes_on_a_beach__Calvin_wearing_a_red.png)
![](man/figures/20250601_111817_Beach_at_night__glowing_waves__stars_in_the_sky__h.png)

## Installation

First install torch. As per [this comment](https://github.com/mlverse/torch/issues/1198#issuecomment-2419363312), using the pre-built binaries from [https://torch.mlverse.org/docs/articles/installation#pre-built](https://torch.mlverse.org/docs/articles/installation#pre-built) is heavily recommend. "The pre-built binaries bundle the necessary CUDA and cudnn versions, so you don't need a global compatible system version of CUDA":

```r
options(timeout = 600) # increasing timeout is recommended since we will be downloading a 2GB file.
# For Windows and Linux: "cpu", "cu124" are the only currently supported
# For MacOS the supported torch versions are: "cpu-intel" or "cpu-m1"
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
devtools::install_github("cornball-ai/diffuseR")
# Or
# install.packages("targets")
targets::install_github("cornball-ai/diffuseR")
```

## Features

- **Text-to-Image Generation**: Create images from textual descriptions
- **Image-to-Image Generation**: Modify existing images based on text prompts
- **Two Models**: Support for Stable Diffusion 2.1 and SDXL (more coming soon)
- **Scheduler Options**: DDIM (more coming soon)
- **Device Support**: Works on both CPU and GPU
- **R-native Interface**: Functional programming approach that feels natural in R

## Quick Start

### Basic Usage

**Warning**: The first time you run the code below, it will download ~5.3GB of Stable Diffusion 2.1 CPU-only model files [from Hugging Face](https://huggingface.co/datasets/cornball-ai/sd21-R/tree/main) and load them into memory. Ensure you have enough RAM, disk space, and a stable internet connection. Memory management with deep learning models is crucial, so consider using a machine with sufficient resources; ~8GB of free RAM is recommended for running Stable Diffusion 2.1 on CPU only.

```r
options(timeout = 600) # increasing timeout is recommended since we will be downloading a 3.5GB file.
library(diffuseR)
torch::local_no_grad()

# Generate an image from text
cat_img <- txt2img(
  prompt = "a photorealistic cat wearing sunglasses",
  model = "sd21", # Specify the model to use, e.g., "sd21" for Stable Diffusion 2.1
  download_models = TRUE, # Automatically download the model if not already present
  steps = 30,
  seed = 42,
  filename = "cat.png",
)

# Clear out pipeline to free up GPU memory
pipeline <- NULL
torch::cuda_empty_cache()
```
![](man/figures/cat.png)

### Advanced Usage with GPU

The unet is the most computationally-intensive part of the model, so it is recommended to run it on a GPU if possible. The decoder and text encoder can be run on CPU if you have limited GPU memory. SDXL's unet requires a minimum of 6GB of GPU memory (VRAM), while Stable Diffusion 2.1 requires a minimum of 2GB.

```r
# Increasing timeout is recommended since we will be downloading 5.1 and 2.8GB model files, among others.
options(timeout = 1200) 

library(diffuseR)
torch::local_no_grad() # Prevents torch from tracking gradients, which is not needed for inference

# Assign the various deep learning models to devices
model_name = "sdxl"
devices = list(unet = "cuda", decoder = "cpu",
               text_encoder = "cpu", encoder = "cpu")

m2d <- models2devices(model_name = model_name, devices = devices,
                      unet_dtype_str = "float16", download_models = TRUE)

pipeline <- load_pipeline(model_name = model_name, m2d = m2d, i2i = TRUE,
                          unet_dtype_str = "float16")

# Generate an image from text
cat_img <- txt2img(
  prompt = "a photorealistic cat wearing sunglasses",
  model_name = model_name,
  devices = devices,
  pipeline = pipeline,
  num_inference_steps = 30,
  guidance_scale = 7.5,
  seed = 42,
  filename = "cat2.png",
)

gambling_cat <- img2img(
  input_image = "cat2.png",
  prompt = "a photorealistic cat throwing dice",
  img_dim = 1024,
  model_name = model_name,
  devices = devices,
  pipeline = pipeline,
  num_inference_steps = 30,
  strength = 0.75,
  guidance_scale = 7.5,
  seed = 42,
  filename = "gambling_cat.png"
)

# Clear out pipeline to free up GPU memory
pipeline <- NULL
torch::cuda_empty_cache()
```
![](man/figures/cat2.png)
![](man/figures/gambling_cat.png)

## Supported Models

Currently supported models:

- Stable Diffusion 2.1
- Stable Diffusion XL (SDXL)
- More coming soon!

## Roadmap

Future plans for diffuseR include:

- [ ] Inpainting support
- [ ] Additional schedulers (PNDM, DPMSolverMultistep, Euler ancestral)
- [ ] text-to-video generation

## How It Works

diffuseR uses TorchScript models exported from PyTorch implementations for the deep learning parts of the implementation. This approach was the quickest and easiest way to build the machinery that supports the deep learning models in stable diffusion. Full R torch implementations of the deep learning models are planned for the future, but this initial version allows users to quickly get started with diffusion models in R without needing to rely on any Python.

If you would like to build the TorchScript (.pt) files yourself, those scripts are available at the [diffuseR-TS](https://github.com/cornball-ai/diffuseR-TS/) repository. The models are exported from the Hugging Face diffusers library.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache 2. License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for the original diffusers library
- Stability AI for Stable Diffusion
- The R and torch communities for their excellent tooling
