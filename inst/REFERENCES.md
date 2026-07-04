# LTX-2.3 implementation references

Every technique in diffuseR's LTX-2.3 support traces to a public,
permissively licensed source or to our own measured engineering. None of
it derives from Wan2GP (WanGP Community License) or its `mmgp` module;
this file documents the actual lineage, idea by idea.

## Model architecture and pipeline

| What | Source |
|---|---|
| DiT, VAEs, connectors, vocoder, upsampler module code | Ported from HuggingFace **diffusers** (Apache-2.0): `models/transformers/transformer_ltx2.py`, `models/autoencoders/autoencoder_kl_ltx2.py`, `autoencoder_kl_ltx2_audio.py`, `pipelines/ltx2/{connectors,vocoder,latent_upsampler}.py` |
| Pipeline flow, Euler velocity steps, latent packing, AdaIN filter, tone mapping | diffusers `pipelines/ltx2/pipeline_ltx2.py`, `pipeline_ltx2_latent_upsample.py` (Apache-2.0) |
| Distilled sigma schedules, default negative prompt | diffusers `pipelines/ltx2/utils.py` (Apache-2.0); values originate from Lightricks' LTX-2 release (numeric facts) |
| Single-file checkpoint key layout, embedded config, `model_version` metadata | Format facts of the official Lightricks checkpoints, cross-checked against diffusers `scripts/convert_ltx2_to_diffusers.py` (Apache-2.0) |
| Gemma3 text encoder | Ported from HuggingFace **transformers** (Apache-2.0) |

## Quantization

| What | Source |
|---|---|
| FP8 e4m3fn weights with per-tensor scales, upcast-in-forward; the exact linear cast set (attention + FFN projections in transformer blocks) | Official Lightricks LTX-2 quantization policy (design facts from `ltx-core/quantization/fp8_cast.py`, LTX-2 Community License — policy observed, no code taken); fp8 storage-with-upcast is also diffusers' documented "layerwise casting" (Apache-2.0) |
| NF4 4-bit format: 16-level NormalFloat quantile code, per-64-block absmax, packed nibbles | **QLoRA**: Dettmers, Pagnoni, Holtzman, Zettlemoyer (2023), "QLoRA: Efficient Finetuning of Quantized LLMs", arXiv:2305.14314; format details as implemented in bitsandbytes (MIT) — reimplemented here in pure torch ops |
| 4-bit weights for consumer-GPU video DiTs as a practice | Community standard: GGUF Q4 checkpoints for LTX/ComfyUI (e.g. city96/ComfyUI-GGUF ecosystem), diffusers bitsandbytes integration docs |

## Memory management

| What | Source |
|---|---|
| Phase-sequential component offloading (each component on the GPU only for its phase) | diffusers memory docs: `enable_model_cpu_offload`, group offloading (`docs/source/en/optimization/memory.md`, Apache-2.0) |
| CPU-resident weights streamed per layer | diffusers leaf-level/group offloading with pinned memory (same doc); long-standing technique (e.g. DeepSpeed ZeRO-Offload, arXiv:2101.06840) |
| Tiled VAE decode (overlapping tiles, crossfaded seams; spatial + temporal) | Direct port of diffusers `AutoencoderKLLTX2Video.tiled_decode` / `_temporal_tiled_decode` (Apache-2.0); tiled VAE decoding dates to the original Stable Diffusion AutoencoderKL (`enable_tiling`) |
| Attention query chunking under a memory budget | diffusers attention slicing (`enable_attention_slicing`, Apache-2.0); memory-efficient attention lineage: Rabe & Staats (2021), arXiv:2112.05682 |
| Pinned host memory + non-blocking transfers | PyTorch CUDA semantics documentation |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` | PyTorch CUDA caching-allocator documentation |
| R-level allocator GC tuning (`torch.cuda_allocator_reserved_rate`, `torch.threshold_call_gc`) | mlverse torch memory-management article (`vignettes/articles/memory-management.Rmd`); A/B methodology and measurements from our own whisper/chatterbox work (both public cornball-ai packages) |
| Persistent scratch buffers for dequantization and attention temporaries | Our own engineering, driven by R's lazy garbage collection of tensor handles (documented in the mlverse article above) |
| Per-block `gc()` for streamed-weight temporaries | Our own engineering (see whisper/chatterbox inference loops) |

## Measured on our hardware (RTX 5060 Ti 16GB)

Resolution caps, VRAM peaks, profile thresholds, and the GC A/B results
in `tasks/todo.md` and commit messages are our own measurements of the
above techniques, not third-party numbers.
