# TorchScript Migration Plan

## Problem

TorchScript (.pt files) are brittle across GPU generations. Blackwell GPUs (RTX 50xx) fail when all components run on CUDA. This requires the `unet_gpu` workaround.

## Goal

Replace TorchScript with native R torch modules that load weights directly from safetensors/HuggingFace.

## Prior Work

**~/sd-rtorch-old** contains native torch implementations:
- `R/VAE.R` - AutoencoderKL with encoder/decoder
- `R/UNet.R` - Full UNet with attention, spatial transformers
- `R/ClipTextEncoder.R` - CLIP text encoder
- `R/load_stable_diffusion_checkpoint.R` - Weight loading from safetensors

The blocker was **weight mapping** - matching checkpoint parameter names to R model parameter names.

## Architecture

```
Current (TorchScript):
  HuggingFace → Python export → .pt files → torch::jit_load()

Target (Native):
  HuggingFace → safetensors → torch::load_state_dict() → R nn_module
```

## Component Priorities

1. **VAE Decoder** (smallest, fastest win)
   - 190MB, simple architecture
   - Most likely to "just work" on Blackwell

2. **Text Encoders** (medium complexity)
   - text_encoder: 470MB (CLIP ViT-L)
   - text_encoder2: 2.6GB (OpenCLIP ViT-bigG)

3. **UNet** (largest, most complex)
   - 4.8GB float16
   - Complex architecture with attention, cross-attention, timestep embedding

## Implementation Steps

### Phase 1: VAE Decoder
1. Port `VAE.R` from sd-rtorch-old to diffuseR
2. Download decoder safetensors from HuggingFace
3. Map weight names (Python → R parameter names)
4. Test decode on Blackwell with full CUDA
5. Benchmark vs TorchScript

### Phase 2: Text Encoders
1. Port `ClipTextEncoder.R` to diffuseR
2. Handle both CLIP models (text_encoder, text_encoder2)
3. Weight mapping for transformer layers
4. Test on Blackwell

### Phase 3: UNet
1. Port `UNet.R` and supporting modules
2. This is the big one - attention blocks, spatial transformers
3. Weight mapping is complex (input_blocks, output_blocks, etc.)
4. Test full pipeline on Blackwell

### Phase 4: Cleanup
1. Remove TorchScript download code
2. Update model download to fetch safetensors
3. Remove Blackwell workaround from gpuctl
4. Update documentation

## Weight Mapping Strategy

The main challenge is mapping HuggingFace parameter names to R module names.

Example VAE mapping:
```
HuggingFace                          R Module
-----------                          --------
decoder.conv_in.weight        →      decoder$conv_in$weight
decoder.mid.block_1.norm1     →      decoder$mid$block_1$norm1
```

Strategy:
1. Load safetensors, print all keys
2. Print R model parameter names
3. Build mapping table (can be automated with pattern matching)
4. Use `model$parameters[[name]]$copy_(tensor)` to load

## Resources

- sd-rtorch-old repo: `~/sd-rtorch-old/`
- HuggingFace SDXL: `stabilityai/stable-diffusion-xl-base-1.0`
- safetensors format: https://huggingface.co/docs/safetensors

## Progress

### Phase 1: VAE Decoder - COMPLETE
- [x] Native `vae_decoder_native()` module created
- [x] `load_decoder_weights()` loads from TorchScript .pt files
- [x] Equivalency verified: max diff 2e-5 (float precision)
- [x] 138/138 parameters match
- [ ] CUDA test pending (need VRAM - ollama currently using GPU)

Files added:
- `R/vae_decoder.R` - Native decoder + weight loader

## Success Criteria

- [ ] Full SDXL pipeline runs on Blackwell with all components on CUDA
- [ ] No TorchScript files required
- [ ] Performance equal or better than TorchScript
- [ ] gpuctl Blackwell workaround can be removed
