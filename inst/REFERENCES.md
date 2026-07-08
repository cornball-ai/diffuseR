# Implementation references (LTX-2.3, FLUX.1-schnell)

Every technique in diffuseR's LTX-2.3 and FLUX.1 support traces to a
public, permissively licensed source or to our own measured engineering.
None of it derives from Wan2GP (WanGP Community License) or its `mmgp`
module; this file documents the actual lineage, idea by idea.

## Model architecture and pipeline

| What | Source |
|---|---|
| DiT, VAEs, connectors, vocoder, upsampler module code | Ported from HuggingFace **diffusers** (Apache-2.0): `models/transformers/transformer_ltx2.py`, `models/autoencoders/autoencoder_kl_ltx2.py`, `autoencoder_kl_ltx2_audio.py`, `pipelines/ltx2/{connectors,vocoder,latent_upsampler}.py` |
| Pipeline flow, Euler velocity steps, latent packing, AdaIN filter, tone mapping | diffusers `pipelines/ltx2/pipeline_ltx2.py`, `pipeline_ltx2_latent_upsample.py` (Apache-2.0) |
| Distilled sigma schedules, default negative prompt | diffusers `pipelines/ltx2/utils.py` (Apache-2.0); values originate from Lightricks' LTX-2 release (numeric facts) |
| Single-file checkpoint key layout, embedded config, `model_version` metadata | Format facts of the official Lightricks checkpoints, cross-checked against diffusers `scripts/convert_ltx2_to_diffusers.py` (Apache-2.0) |
| Gemma3 text encoder | Ported from HuggingFace **transformers** (Apache-2.0) |

## FLUX.1-schnell

| What | Source |
|---|---|
| MMDiT transformer (double/single blocks, joint attention, adaLN-Zero variants, RoPE position ids, timestep + pooled-text conditioning) | Ported from HuggingFace **diffusers** (Apache-2.0): `models/transformers/transformer_flux.py`, `models/normalization.py`, `models/embeddings.py` |
| Pipeline flow: prompt encoding contract, sigma schedule (`linspace(1, 1/N, N)`, static shift), 2x2 latent pack/unpack, latent image ids, VAE scale/shift decode | diffusers `pipelines/flux/pipeline_flux.py` (Apache-2.0) |
| FlowMatch Euler scheduler | diffusers `schedulers/scheduling_flow_match_euler_discrete.py` (Apache-2.0); shared with the LTX port |
| 16-channel AutoencoderKL decoder config (no quant convs, scaling 0.3611 / shift 0.1159) | diffusers `scripts/convert_flux_to_diffusers.py` + `convert_sd3_to_diffusers.py` (Apache-2.0) |
| T5-v1.1 encoder (RMS norms, unscaled unbiased attention, shared relative position bias, gated-GELU FFN) | Ported from HuggingFace **transformers** (Apache-2.0): `models/t5/modeling_t5.py` |
| CLIP ViT-L text encoder with quick-GELU and argmax-EOS pooling | HuggingFace **transformers** `models/clip/modeling_clip.py` (Apache-2.0); native module shared with the SD/SDXL port |
| SentencePiece Unigram tokenization (Viterbi best-path over piece log-probs) | Kudo (2018), "Subword Regularization", arXiv:1804.10959; SentencePiece (Apache-2.0); tokenizer.json format facts from HuggingFace tokenizers documentation |
| NF4/fp8 transformer quantization, cast-set policy, phase offloading, allocator tuning | Same sources as the LTX sections below, applied to the FLUX cast set |
| Weights | black-forest-labs/FLUX.1-schnell (Apache-2.0; gated HuggingFace repo, downloaded by the user, never redistributed) |

## FLUX.2 klein

| What | Source |
|---|---|
| MMDiT transformer (shared modulation, SwiGLU feed-forwards, ViT-22B parallel single blocks, 4-axis RoPE) | Ported from HuggingFace **diffusers** (Apache-2.0): `models/transformers/transformer_flux2.py` |
| Klein pipeline flow: Qwen3 prompt encoding contract (chat template, mid-stack hidden states 9/18/27), 4-channel position ids, latent patchify/pack chain, empirical dynamic-shift mu, BatchNorm latent statistics | diffusers `pipelines/flux2/pipeline_flux2_klein.py` (Apache-2.0); the empirical mu formula originates from BFL `sampling.py` (numeric facts) |
| 32-channel `AutoencoderKLFlux2` decode path (post_quant_conv + standard decoder + BN running stats) | diffusers `models/autoencoders/autoencoder_kl_flux2.py` (Apache-2.0); decoder body shared with the SD/FLUX.1 port |
| Qwen3 encoder (GQA, per-head q/k RMS norms, SwiGLU, RoPE theta 1e6, causal + padding mask) | Ported from HuggingFace **transformers** (Apache-2.0): `models/qwen3/` |
| Qwen2 byte-level BPE tokenization (GPT-2 byte table, GPT-4-style split regex, rank-based merges) | GPT-2 BPE (Radford et al. 2019; openai/gpt-2 `encoder.py`, MIT); format facts from HuggingFace tokenizers documentation |
| Weights | black-forest-labs/FLUX.2-klein-4B (Apache-2.0, ungated; downloaded by the user, never redistributed) |

## Z-Image Turbo

| What | Source |
|---|---|
| Single-stream DiT (sandwich RMSNorms, scale/gate-only tanh modulation, noise/context refiner stacks, SwiGLU with separate gate, 3-axis RoPE theta 256, SEQ_MULTI_OF padding with learned pad tokens) | Ported from HuggingFace **diffusers** (Apache-2.0): `models/transformers/transformer_z_image.py` |
| Pipeline flow: Qwen3 prompt encoding contract (thinking-enabled chat template, penultimate hidden state, mask-sliced captions), linspace sigma schedule, reversed-timestep + negated-output convention, static shift | diffusers `pipelines/z_image/pipeline_z_image.py` (Apache-2.0) |
| VAE decode | The FLUX.1 16-channel AutoencoderKL port, unchanged (the checkpoint's VAE config is the flux-dev autoencoder) |
| Qwen3-4B encoder + Qwen2 byte-level BPE tokenizer | Shared with the FLUX.2 klein port (see above); the tokenizer files are byte-identical between the two checkpoints |
| Weights | Tongyi-MAI/Z-Image-Turbo (Apache-2.0, ungated; downloaded by the user, never redistributed) |

## Quantization

| What | Source |
|---|---|
| FP8 e4m3fn weights with per-tensor scales, upcast-in-forward; the exact linear cast set (attention + FFN projections in transformer blocks) | Official Lightricks LTX-2 quantization policy (design facts from `ltx-core/quantization/fp8_cast.py`, LTX-2 Community License — policy observed, no code taken); fp8 storage-with-upcast is also diffusers' documented "layerwise casting" (Apache-2.0) |
| NF4 4-bit format: 16-level NormalFloat quantile code, per-64-block absmax, packed nibbles | **QLoRA**: Dettmers, Pagnoni, Holtzman, Zettlemoyer (2023), "QLoRA: Efficient Finetuning of Quantized LLMs", arXiv:2305.14314; format details as implemented in bitsandbytes (MIT) — reimplemented here in pure torch ops |
| 4-bit weights for consumer-GPU video DiTs as a practice | Community standard: GGUF Q4 checkpoints for LTX/ComfyUI (e.g. city96/ComfyUI-GGUF ecosystem), diffusers bitsandbytes integration docs |

## Conditioning (image-to-video, continuation, audio-driven)

| What | Source |
|---|---|
| Prefix frame conditioning: VAE-encoded pixels as frozen latent tokens, per-token timestep `t * (1 - mask)`, frozen Euler updates, partial-noise option | Ported from diffusers `pipelines/ltx2/pipeline_ltx2_image2video.py` and `pipeline_ltx2_condition.py` (Apache-2.0), restricted to index-0 / strength-1 |
| Audio-driven generation (lip sync): user audio encoded to clean, frozen audio latents; audio stream sees timestep/sigma zero while video denoises attending to it | Our own design — the audio-side mirror of the Apache i2v conditioning pattern above (diffusers has no audio-input conditioning) |
| Audio VAE encoder module | Ported from diffusers `autoencoder_kl_ltx2_audio.py` `LTX2AudioEncoder` (Apache-2.0) |
| 16 kHz log-mel frontend (causal STFT 1024/160, 64 slaney-normed mel bins to 8 kHz) | Parameters are the checkpoint's embedded preprocessing config (numeric facts); STFT/mel constructions are textbook DSP, verified against the checkpoint's stored vocoder bases |

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
| TorchScript-compiled block stack (`torch::jit_compile`, weights as `List[Tensor]`, fused `scaled_dot_product_attention`) | mlverse torch JIT API; the JIT-decode pattern from our own whisper (`decode_jit.R`) and chatterbox (`t3_jit.R`) work, both public cornball-ai packages |

## Measured on our hardware (RTX 5060 Ti 16GB)

Resolution caps, VRAM peaks, profile thresholds, and the GC A/B results
in `tasks/todo.md` and commit messages are our own measurements of the
above techniques, not third-party numbers.
