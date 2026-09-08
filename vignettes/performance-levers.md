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
it to pick between built LTX artifacts; the FLUX-family loaders resolve
their `pin` (and FLUX.1 its `text_device`) through it. For the rest of
the fields it is advisory — call it and pass them to the loaders
yourself.

## Lever 1: weight precision

The ladder that exists in code: fp32 → bf16/fp16 → fp8 (e4m3fn) → nf4.
Quantization applies to the big diffusion transformers only — VAEs,
vocoders, connectors, and modulation/embedding layers always stay at
16/32 bits (they are small and precision-sensitive; each quantizer
carries an exact census of which weights it may touch).

| model | DiT / UNet | text encoder(s) | VAE |
|---|---|---|---|
| FLUX.1 (12B) | nf4, fp8 (streamed), bf16, fp32 | T5: bf16 (GPU, 14 GB+) or fp32 (CPU) · CLIP-L: fp16/fp32 | 16/32 |
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
the loop. Text encoders earn special placement: FLUX.1's T5
phase-onloads in bf16 on 14 GB+ cards (its ~9.8 GB encode phase fits)
and runs fp32 on the CPU below that; the Qwen3 encoders phase-onload in
bf16; the Gemma3 encoder can be GPU-resident or CPU-resident with a
staged swap (see below).

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
   ~0.6 s/GB once at load. Built for the LTX pipeline components, the
   Gemma3 encoder, and the FLUX-family transformer, VAE decoder, and
   text encoder(s).
4. **Pageable phase swap** — the same movement through ordinary
   memory, at roughly 2-16 GB/s depending on tensor layout. The
   fallback when pinning is opted out or page-locking fails.
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
itself back. The LTX pipeline, the Gemma3 loaders, and the FLUX-family
image loaders consume the decision (their `pin` argument defaults to
it); the SD-family loaders place components statically, so pinning is
inert for them.

## Putting it together

```r
r <- recommend("ltx")     # or "flux1", "flux2", "zimage", "sdxl", "sd21"
r$precision               # tier the card + safetensors support
r$devices                 # per-component placement
r$pin                     # page-lock the phase-swapped host copies?
r$note                    # upgrade suggestion when fp8 wanted but unreadable
```

Treat the result as the machine's advice: pass its fields to the
loaders and generators. The FLUX-family loaders' `pin` (and FLUX.1's
`text_device`), `flux_memory_profile()`, and `serve()`'s LTX artifact
selection consume it automatically.

## Where the time goes: TorchScript and the allocator

The levers above decide what fits. A second set decides how fast the
fitted configuration runs. They came out of making the 22B NF4 LTX
DiT fast on a 16 GB card, where a same-seed render went from 1331 s
to 221 s without changing a weight.

### Compile the hot step

R torch pays twice per eager op: the dispatch itself and the R handle
each intermediate becomes, which the garbage collector then has to
sweep. `jit_compile()` removes both. The LTX denoiser step compiles
into one TorchScript call: a `while` loop over the 48 blocks inside
the script, weights passed as a flat `List[Tensor]` with a fixed
per-block slot layout, and the NF4 dequant math in-script. Measured on
the same step: eager 99.9 s at 88% of wall time in R's gc, compiled
8.8 s at 1%.

Builtins lantern's TorchScript accepts beyond the documented list:
`torch.scaled_dot_product_attention` (fused, so materialized
`[B,H,S,S]` score matrices and their chunked-attention workarounds go
away), `torch.gelu(x, approximate="tanh")`, `torch.sigmoid`,
`torch.silu`, `torch.rsqrt`, `torch.linear`,
`torch.bitwise_right_shift`, `torch.bitwise_and`, `torch.index_select`,
and the `.long()` method cast. That set is enough for NF4 dequant
fully in-script; chunk the loop to bound the int64 index temporaries.

Marshalling traps, all silent or opaque:

- A named R list marshals as `Dict[str, Tensor]`, not `List[Tensor]`.
  `lapply()` over an `nn_module_list` yields named children, so
  `unname()` the packed weight list. Parity-test the packer against
  the script's slot indices.
- R integers marshal as `List[int]`; wrap scalars in `jit_scalar()`.
- Dtype constants such as `torch.long` do not resolve; use method
  casts and `type_as()`.
- `Optional[Tensor]` parameters accept R `NULL`. Cross-def calls
  within one compilation unit work. Tuple returns come back as R
  lists.
- Compile once per session and cache the compilation unit. Passing
  5000+ tensor references per call costs milliseconds.

### When tracing loses

`jit_trace()` converts static feed-forward modules (VAE decoders,
vocoders) wholesale with exact parity, but it bakes runtime shapes
into the graph, so cache one trace per input shape. Trace a closure
rather than a bare `nn_module`, and set `requires_grad_(FALSE)` on the
parameters first. Two hazards decide whether it pays:

- An R gc during recording corrupts captured argument values. Under
  memory pressure the allocator callback can fire mid-trace and the
  graph records garbage narrow offsets. Run `gc()` and
  `cuda_empty_cache()` immediately before tracing, wrap the trace in
  `tryCatch()`, and validate every fresh trace against the eager
  output once, falling back to eager for that shape on mismatch.
- A trace captures the module's weights and pins them on the device
  across phase offloads, so it must be released when the component
  leaves the GPU. A once-per-render decode then re-pays trace plus
  validation every run: traced was net slower than eager (39.8 s vs
  32.6 s). `jit_compile()` from source is shape-generic and captures
  nothing, so prefer it wherever you can afford to write the script.

### Allocator cost models

`PYTORCH_CUDA_ALLOC_CONF` picks between two cost models, and the right
one depends on whether intermediates still exist as R handles:

- `expandable_segments:True`: cheap growth, expensive frees (page
  unmaps). Suits eager paths with fragmentation churn.
- `backend:native`: expensive growth (about 15 ms per `cudaMalloc`),
  cheap frees. Suits a JIT path that keeps intermediates out of R.

The same decode measured 32.7 s at 86% gc on expandable segments and
21.0 s at 50% on native. The setting is process-wide and read before
the first CUDA allocation, so set it per model at load time.

When a phase is gc-bound, ask where the gc time goes before touching
options. If the trigger frequency is the problem, the callback gates
help: `torch.cuda_allocator_reserved_rate` plus the two most people
forget, `torch.cuda_allocator_allocated_rate` and
`torch.cuda_allocator_allocated_reserved_rate` (both default 0.8).
Raising those two to 0.95 took the expandable-segments decode from
32.7 s to 21.5 s. If the freeing work itself is the problem, no
option helps; the fix is fewer or cheaper frees, through the backend
choice or through JIT so the handles never exist.

torch reads all three gates once, in its `.onLoad`. Any package that
imports torch has it loaded before its own tuner runs, and the set
silently no-ops while `getOption()` still returns the value you set.
`ltx23_tune_gc()` therefore pushes the gates into the live allocator
as well as setting the options.

### Pool regrowth between phases

`cuda_empty_cache()` between pipeline phases returns every block to
the driver, so the next phase regrows the pool one `cudaMalloc` per
tensor. Measured: 5530 tensors, 83.5 s for one 11 GB component
onload, and back to 83 s after a single empty-cache. Two fixes:

- Pre-warm once by allocating one footprint-sized tensor and freeing
  it into the cache (0.2 s, one `cudaMalloc`); later per-tensor
  allocations carve from it (83.5 s to 4.6 s).
- Between phases, `gc()` without `cuda_empty_cache()`. The caching
  allocator reuses freed blocks across phases on its own.

The corollary is that "phase offload traffic" may not be transfer
time at all. Pinned staging is worth having and roughly doubles
transfer rate, but the transfers here were about 1 s all along; the
110 s was regrowth. Time the transfer in isolation before optimizing
it.

### Compact per-token conditioning

Prefix conditioning gives conditioned tokens timestep 0 and free
tokens timestep t, a per-token timestep vector with only a few
distinct values. Materializing per-token modulation as
`[B,S,num_params,D]` cost about 2 GB at 13.5k tokens and ran out of
memory. Passing the distinct timesteps plus an integer index and doing
`index_select` per modulation vector inside the script materializes
one `[B,S,D]` slice at a time (about 110 MB). Whenever a per-token
tensor is an index into few variants, ship the variants and the index
and select late.

### Verify the inputs to tuning formulas

A VRAM-detection helper once returned a hardcoded 8 GB guess when its
optional dependency was missing, and every allocator rate computed
from it was wrong for weeks. If a formula tunes by footprint over
total, log both numbers at tuning time and fail loudly on a fallback
guess. `nvidia-smi` is always there to ask.
