# diffuseR

Native R package for Stable Diffusion image generation.

## Key Functions

| Function | Purpose |
|----------|---------|
| `txt2img_sdxl()` | Generate image from text (SDXL) |
| `txt2img_sd21()` | Generate image from text (SD 2.1) |
| `img2img()` | Image-to-image generation |
| `load_pipeline()` | Pre-load models for reuse |
| `models2devices()` | Configure device placement |

## Usage

```r
library(diffuseR)
torch::with_no_grad({
  # Auto-detect optimal device configuration (requires gpuctl)
  txt2img_sdxl("A sunset over mountains", filename = "output.png")
})
```

With explicit device configuration:
```r
devices <- list(unet = "cuda", decoder = "cpu",
                text_encoder = "cpu", encoder = "cpu")
txt2img_sdxl("A sunset over mountains", devices = devices)
```

## GPU Poor Support

diffuseR integrates with gpuctl for automatic "GPU poor" support:

```r
# Auto-detect VRAM and select optimal strategy
devices <- auto_devices("sdxl")  # Uses gpuctl if available

# Strategies (SDXL thresholds):
# - full_gpu: All on CUDA (10GB+ VRAM)
# - unet_gpu: Unet on CUDA, rest CPU (6GB+ VRAM) - forced on Blackwell
# - cpu_only: All on CPU (<6GB VRAM)
devices <- auto_devices("sdxl", strategy = "unet_gpu")
```

The `txt2img_*` and `img2img` functions default to `devices = "auto"`, which:
1. Uses gpuctl to detect VRAM and GPU architecture
2. Selects optimal strategy (full_gpu, unet_gpu, or cpu_only)
3. Forces unet_gpu on Blackwell GPUs (TorchScript workaround)

## Native Torch Migration (Complete)

Replaced TorchScript with native R torch modules for Blackwell GPU compatibility.

### Completed Components

| Component | Status | Notes |
|-----------|--------|-------|
| VAE Decoder | ✅ Complete | `use_native_decoder = TRUE` |
| Text Encoder | ✅ Complete | `use_native_text_encoder = TRUE`, auto-detects architecture |
| Text Encoder 2 | ✅ Complete | SDXL's OpenCLIP ViT-bigG |
| UNet (SD21) | ✅ Complete | `use_native_unet = TRUE`, 686 parameters |
| UNet (SDXL) | ✅ Complete | `use_native_unet = TRUE`, 1680 parameters, variable transformer depths |

### Usage with Native Components

```r
# Full native pipeline (works on Blackwell)
txt2img_sd21("A cat wearing a hat",
             use_native_decoder = TRUE,
             use_native_text_encoder = TRUE,
             use_native_unet = TRUE)

# For SDXL
txt2img_sdxl("A sunset over mountains",
             use_native_decoder = TRUE,
             use_native_text_encoder = TRUE,
             use_native_unet = TRUE)
```

### Architecture Auto-Detection

The native text encoder auto-detects model architecture:

| Model | embed_dim | layers | heads | Final LN |
|-------|-----------|--------|-------|----------|
| SD21 (OpenCLIP ViT-H) | 1024 | 23 | 16 | Yes |
| SDXL text_encoder (CLIP ViT-L) | 768 | 12 | 12 | No |
| SDXL text_encoder2 (OpenCLIP ViT-bigG) | 1280 | 32 | 20 | No |

See `TORCHSCRIPT_MIGRATION.md` for detailed migration progress.

## Critical: Inference Memory Management

**ALWAYS use `torch::with_no_grad()` for inference.** Without it, PyTorch builds computation graphs that consume massive amounts of VRAM.

The `txt2img_*` functions now wrap the denoising loop in `with_no_grad()`. If writing custom inference code:

```r
# CORRECT - no gradient tracking
torch::with_no_grad({
  output <- unet(latents, timestep, prompt_embed)
})

# WRONG - will OOM on large models
output <- unet(latents, timestep, prompt_embed)
```

## Known Issues / TODOs

### Blackwell GPU Compatibility

**Status**: Resolved with native modules.

All components now have native R torch implementations that work on Blackwell GPUs (RTX 50xx):
- Use `use_native_unet = TRUE`, `use_native_decoder = TRUE`, `use_native_text_encoder = TRUE`

### Model Files

Models stored in `~/.local/share/R/diffuseR/{model_name}/`:
- `unet-{device}-{dtype}.pt` (TorchScript - still required)
- `decoder-{device}.pt` (TorchScript - used for weight loading)
- `text_encoder-{device}.pt` (TorchScript - used for weight loading)
- `encoder-{device}.pt` (TorchScript)

Downloaded from: `huggingface.co/datasets/cornball-ai/sdxl-R`

## Roadmap

### Native Torch Implementation (Complete)
- [x] **VAE Decoder**: Native implementation complete
- [x] **Text Encoder**: Native implementation with auto-detection
- [x] **Text Encoder 2**: OpenCLIP ViT-bigG for SDXL
- [x] **UNet SD21**: 4 blocks, 686 parameters
- [x] **UNet SDXL**: 3 blocks, variable transformer depths (0, 2, 10), 1680 parameters

### gpuctl Integration
- [x] **Auto-device configuration**: `auto_devices()` integrates with gpuctl
  - Queries available VRAM via `gpuctl::gpu_detect()`
  - Auto-selects optimal devices based on model requirements
  - Handles Blackwell workaround automatically

### Model Support
- [ ] Add FLUX model support
- [ ] Add SD3 model support
- [ ] ControlNet integration
