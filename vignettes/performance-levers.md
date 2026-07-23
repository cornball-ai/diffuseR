<!--
%\VignetteIndexEntry{Performance Levers}
%\VignetteEngine{simplermarkdown::mdweave_to_html}
%\VignetteEncoding{UTF-8}
-->
---
title: "Performance Levers"
---

# Performance Levers

diffuseR runs 20B-parameter models on 16 GB consumer GPUs, and most
models on no GPU at all. It does that with three independent levers:
weight precision, per-component device placement, and memory residency.
This vignette is the map. The machine-readable version of the same
policy is `recommend()`, which inspects your VRAM, host RAM, and
installed `safetensors` capabilities and returns a configuration.
Today `flux_memory_profile()` delegates to it and `serve()` consults
it to pick between built LTX artifacts; for everything else it is
advisory — call it and pass its fields to the loaders yourself. Full
consumption (including `pin`) is arriving model by model.

## Lever 1: weight precision

The ladder that exists in code: fp32 → bf16/fp16 → fp8 (e4m3fn) → nf4.
Quantization applies to the big diffusion transformers only — VAEs,
vocoders, connectors, and modulation/embedding layers always stay at
16/32 bits (they are small and precision-sensitive; each quantizer
carries an exact census of which weights it may touch).

| model | DiT / UNet | text encoder(s) | VAE |
|---|---|---|---|
| FLUX.1 (12B) | nf4, fp8 (streamed), bf16, fp32 | T5: fp32 (CPU) · CLIP-L: fp16/fp32 | 16/32 |
| FLUX.2 klein (4B) | nf4, fp8 (resident), bf16, fp32 | Qwen3-4B: bf16/fp32 | 16/32 |
| Z-Image (6B) | nf4, fp8 (resident), bf16, fp32 | Qwen3-4B: bf16/fp32 | 16/32 |
| LTX-2.3 (22B video) | nf4, fp8 (streamed), bf16, fp32 | Gemma3-12B: nf4, bf16, fp32 | 16/32 |
| SD 2.1 / SDXL | fp16, fp32 | CLIP: fp16/fp32 | 16/32 |

Two readability rules govern the ladder:

- **nf4 always loads.** Its artifacts are packed uint8 plus float32
  scale blocks in sub-2 GB shards — every `safetensors` build reads
  them. It is the default tier for the quantized families.
- **fp8 needs a capable `safetensors`.** The float8 dtypes are not yet
  readable by the CRAN `safetensors`; `recommend()` probes the
  installed build and, when a card could run fp8 but the reader
  cannot, recommends nf4 and surfaces the suggestion in `$note`
  (never an error).

The SD-family models ship no quantized weights: their floor is fp16,
and what varies across VRAM is placement, not precision.

## Lever 2: device placement

Every component takes its own device. The SD family uses explicit
device maps (`auto_devices()` strategies: `full_gpu`, `unet_gpu`,
`cpu_only`); the flux family and LTX use phase offloading, where each
component holds the GPU only for its own phase — text encoding,
denoising, decoding — and the denoiser is the sole GPU tenant during
the loop. Text encoders earn special placement: FLUX.1's T5 runs
fp32 on the CPU for quality; the Qwen3 encoders phase-onload in bf16;
the Gemma3 encoder can be GPU-resident or CPU-resident with a staged
swap (see below).

## Lever 3: residency

From most to least VRAM:

1. **Fully GPU-resident** — everything fits, nothing moves (large
   cards only).
2. **Resident quantized DiT, phase-swapped everything else** — the
   16 GB sweet spot: LTX nf4 and the fp8 FLUX.2/Z-Image
   configurations keep the DiT on the card while text encoders swap
   per phase.
3. **Pinned phase swap** — component weights live in page-locked
   ("pinned") host RAM and move to the GPU for their phase at DMA
   rate. Measured on a PCIe 5.0 x8 card: 25 GB/s host-to-device
   (11 GB of transformer re-onloads in 0.5 s), and the offload is a
   pointer swap back to the still-valid pinned copy — zero bytes
   moved, because inference never mutates weights. Page-locking costs
   ~0.6 s/GB once at load. Built today for the LTX pipeline
   components and the Gemma3 encoder.
4. **Pageable phase swap** — the same movement through ordinary
   memory, at roughly 2-16 GB/s depending on tensor layout. What the
   flux-family encoders do today.
5. **Streamed weights** — the bigger-than-VRAM tier: weights stay in
   (pinned) host RAM permanently and stream across PCIe during each
   forward pass, about one byte per parameter per step. LTX fp8 and
   FLUX.1 fp8 run this way.
6. **CPU-only** — every model runs without a GPU, at its floor
   precision.

There is no disk tier at inference time: weights load from disk once,
and the closest thing to "swap to disk" is *unpinned* host memory
being paged out by the OS — which is exactly the trade `recommend()`
weighs.

## Pinning: when and when not

Pinned pages are unswappable — they subtract from what the OS can page
out, so on small-RAM machines they convert memory pressure into
process kills rather than slowdowns. `recommend()` therefore returns
`pin = TRUE` only when available host RAM covers the model's estimated
pinned set twice over, `FALSE` on the CPU tier (nothing stages), and
`TRUE` when RAM cannot be detected, because page-locking already fails
soft per component. The global switch is
`options(diffuseR.pin_staging = FALSE)` — reach for it under host
memory pressure, in containers with hard memory caps, or for
single-generation sessions where the one-time page-lock never pays
itself back. Today the LTX pipeline and Gemma3 loaders consume the
decision; the image-model loaders do not stage weights yet.

## Putting it together

```r
r <- recommend("ltx")     # or "flux1", "flux2", "zimage", "sdxl", "sd21"
r$precision               # tier the card + safetensors support
r$devices                 # per-component placement
r$pin                     # page-lock the phase-swapped host copies?
r$note                    # fork suggestion when fp8 wanted but unreadable
```

Treat the result as the machine's advice: pass its fields to the
loaders and generators (only `flux_memory_profile()` and `serve()`'s
LTX artifact selection consume it automatically today).
