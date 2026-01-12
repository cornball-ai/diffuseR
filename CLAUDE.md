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

## GPU-Poor Execution Plan (TODO)

Profile-based memory optimization for constrained GPUs. Inspired by mmgp/Wan2GP approach.

### API

```r
txt2img_sdxl("A cat", profile = "auto")  # Default: auto-detect via gpuctl
txt2img_sdxl("A cat", profile = "gpu_poor")  # Force low-memory mode
txt2img_sdxl("A cat", profile = "full_gpu", vram_debug = TRUE)  # Debug VRAM usage
```

### Profiles

| Profile | VRAM | Devices | CFG Mode | Cleanup | Use Case |
|---------|------|---------|----------|---------|----------|
| `full_gpu` | 16GB+ | All CUDA | batched | none | RTX 4090, 5080, 5090 |
| `balanced` | 10-12GB | UNet+decoder CUDA | batched | phase | RTX 3080, 4070 Ti |
| `gpu_poor` | 6-8GB | UNet CUDA only | sequential | phase | RTX 3060, 4060 |
| `extreme` | <6GB | UNet CUDA only | sequential | step | GTX 1660, laptops |
| `cpu_only` | 0 | All CPU | sequential | none | No GPU / testing |

### Profile Details

**full_gpu** (16GB+)
- All components on CUDA
- Batched CFG (uncond+cond in single forward pass)
- No cleanup overhead
- Fastest execution

**balanced** (10-12GB)
- UNet and decoder on CUDA, text encoders on CPU
- Batched CFG
- Phase cleanup between denoise and decode
- Good speed/memory tradeoff

**gpu_poor** (6-8GB)
- Only UNet on CUDA, everything else CPU
- Sequential CFG (halves peak activation memory)
- Phase cleanup + UNet→CPU swap before decode
- Slower but fits in limited VRAM

**extreme** (<6GB)
- Only UNet on CUDA
- Sequential CFG
- Step-level cleanup (gc + cuda_empty_cache each step)
- Slowest but minimum peak memory

### Implementation Components

#### 1. Profile Resolution

```r
resolve_profile <- function(profile = "auto", model = "sdxl") {
  if (profile == "auto") {
    vram <- gpuctl::gpu_detect()$vram_gb
    profile <- if (vram >= 16) "full_gpu"
               else if (vram >= 10) "balanced"
               else if (vram >= 6) "gpu_poor"
               else if (vram > 0) "extreme"
               else "cpu_only"
  }
  get_profile_config(profile, model)
}
```

#### 2. Sequential CFG Mode

Run uncond and cond UNet passes separately instead of batched:

```r
if (cfg_mode == "sequential") {
  noise_pred_uncond <- unet(latents, t, empty_prompt, ...)
  noise_pred_cond <- unet(latents, t, prompt, ...)
  noise_pred <- noise_pred_uncond + scale * (noise_pred_cond - noise_pred_uncond)
  rm(noise_pred_uncond, noise_pred_cond)  # Free immediately
}
```

Halves peak activation memory during UNet forward pass.

#### 3. Phase Cleanup

Clean up between denoise and decode phases:

```r
if (cleanup %in% c("phase", "step")) {
  # Swap UNet to CPU before decode
  pipeline$unet$to(device = "cpu")
  rm(noise_pred, timestep, ...)
  gc()
  torch::cuda_empty_cache()
}
```

#### 4. Step Cleanup (extreme mode)

Most aggressive - cleanup after each denoising step:

```r
for (i in seq_along(timesteps)) {
  # ... denoising step ...
  if (cleanup == "step") {
    rm(noise_pred, timestep)
    gc()
    torch::cuda_empty_cache()
  }
}
```

#### 5. Function Isolation

Extract denoise and decode into separate functions so tensors go out of scope:

```r
run_denoise_sdxl <- function(latents, ...) {
  # Denoise loop
  # Returns ONLY latents (all other tensors die when function exits)
  latents
}

run_decode_sdxl <- function(latents, decoder, device) {
  # Decode latents to image
  # Returns R array (all torch tensors die when function exits)
  img_array
}
```

#### 6. VRAM Debug

```r
vram_report <- function(label) {
  allocated <- torch::cuda_memory_allocated() / 1024^3
  reserved <- torch::cuda_memory_reserved() / 1024^3
  message(sprintf("[%s] VRAM: %.2f GB allocated, %.2f GB reserved",
                  label, allocated, reserved))
}
```

### Future Optimizations (from mmgp)

Not implemented, but worth considering:

- **Pinned CPU memory**: `$pin_memory()` for faster CPU↔GPU transfers
- **Async prefetch**: Load next component while current one computes (needs torch streams)
- **Resident set policy**: Keep hot components on GPU, evict cold ones (useful for batch generation)

## Native Torch Migration (Complete)

Replaced TorchScript with native R torch modules for Blackwell GPU compatibility.

### Component Status

| Component | Status | Notes |
|-----------|--------|-------|
| VAE Decoder | ✅ Complete | `use_native_decoder = TRUE` |
| Text Encoder | ✅ Complete | `use_native_text_encoder = TRUE`, auto-detects architecture |
| Text Encoder 2 | ✅ Complete | SDXL's OpenCLIP ViT-bigG |
| UNet (SD21) | ✅ Complete | `use_native_unet = TRUE`, 686 parameters |
| UNet (SDXL) | ✅ Complete | `use_native_unet = TRUE`, 1680 parameters, fixed timestep_embedding |

### Usage with Native Components

```r
# Full native pipeline for SD21 (works on Blackwell)
txt2img_sd21("A cat wearing a hat",
             use_native_decoder = TRUE,
             use_native_text_encoder = TRUE,
             use_native_unet = TRUE)

# Full native pipeline for SDXL (works on Blackwell)
txt2img_sdxl("A sunset over mountains",
             use_native_decoder = TRUE,
             use_native_text_encoder = TRUE,
             use_native_unet = TRUE)
```

### SDXL Native UNet Fix (January 2026)

The native SDXL UNet initially had ~12% mean error due to incorrect `timestep_embedding()`:

**Root cause:** Model-specific parameters in shared utility function:
- SDXL uses `flip_sin_to_cos=TRUE` (cos before sin), SD21 uses FALSE
- SDXL uses `downscale_freq_shift=0` (divide by half_dim), SD21 uses 1

**Resolution:** Added config parameters to `timestep_embedding()` with correct defaults.
Final output error: <0.1% (within float16 tolerance).

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

**Current (TorchScript - being phased out):**

Models stored in `~/.local/share/R/diffuseR/{model_name}/`:
- `unet-{device}-{dtype}.pt` (TorchScript)
- `decoder-{device}.pt` (TorchScript)
- `text_encoder-{device}.pt` (TorchScript)
- `encoder-{device}.pt` (TorchScript)

Downloaded from: `huggingface.co/datasets/cornball-ai/sdxl-R`

**Future (safetensors):**

Native torch modules will load weights directly from HuggingFace safetensors format:
- No device-specific files needed
- Direct loading from HuggingFace model repos
- Smaller downloads (weights only, no traced graphs)

```r
# Future API (TODO)
model <- load_from_hf("stabilityai/stable-diffusion-xl-base-1.0")
```

See cornyverse CLAUDE.md for safetensors package setup (use cornball-ai fork until PR merged).

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
