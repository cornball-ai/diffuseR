# diffuseR 0.2.0

## Native safetensors pipelines

* SD 2.1, SDXL, FLUX.1-schnell, FLUX.2 Klein, Z-Image-Turbo, and
  LTX-2.3 all run fully natively from diffusers-layout safetensors —
  no TorchScript step, so everything works on Blackwell (RTX 50xx).
  `download_sd21()`/`download_sdxl()` fetch diffusers weights;
  `sd_pipeline_from_safetensors()`/`sdxl_pipeline_from_safetensors()`
  build the pipelines; `txt2img_sd21()`/`txt2img_sdxl()` take
  `diffusers_dir=`.
* Checkpoint loaders build module skeletons (uninitialized weights at
  the target dtype) instead of initializing fp32 and casting: the
  LTX-2.3 NF4 pipeline load drops from ~108 GB host RAM to ~21 GB.
  Loaders hard-error on any parameter the checkpoint does not fill.
* `reshard_safetensors()` splits oversize weights into sub-2 GB shards
  readable by stock CRAN safetensors; quantizer shards default to
  1.9e9 bytes for the same reason. Requesting fp8/bf16 without a
  capable safetensors warns and falls back to nf4; legacy oversize
  shards raise an actionable message.
* Fixed a long-standing native SD 2.1 UNet tiling bug: the timestep
  embedding used the wrong sin/cos ordering
  (`flip_sin_to_cos`/`downscale_freq_shift`), which compounded through
  the spatial path into tiled output. Now matches the TorchScript
  reference at cosine 0.99999.

## Machine-aware configuration

* `recommend(model)`: one VRAM-, host-RAM-, and capability-aware
  recommendation for every model — precision tier, per-component
  device map, phase offload, pixel budget, attention chunking, and
  `pin` (page-lock the phase-swapped host copies; pinned pages are
  unswappable, so pinning is recommended only when available RAM
  covers the model's pinned set twice over). fp8/bf16 tiers are gated
  on the *installed* safetensors' read capabilities; when a card fits
  a tier the reader can't load, the fork suggestion is surfaced in
  `$note`, never as an error.
* The SD tiers are labeled fp16 at every VRAM level: the SD models
  ship no quantized weights, so placement varies, not precision.
* New "Performance Levers" vignette documenting the three axes
  (precision ladder, device placement, memory residency) and a
  hardware-requirements table in the README.

## LTX-2.3 video performance

Warm 768x512x49 renders went from ~90 s to ~44 s across this cycle:

* NF4 dequantization via a precomputed [256, 2] byte lookup table
  (one embedding gather in the compute dtype) instead of an int64
  shift/stack/gather chain — ~6x less per-step memory traffic.
* The CUDA allocator gc gates now actually take effect
  (`ltx23_tune_gc()` used to set its options one call after torch had
  read them) and are pushed into the live allocator.
* Pinned staging for phase offload, on by default: page-locked host
  copies make onload a DMA transfer (25.1 GB/s measured; 11 GB
  re-onloads in 0.5 s) and offload a pointer swap. Page-locked memory
  is allocated via `torch_empty_strided(pin_memory = TRUE)`, avoiding
  a deprecated overload that printed two warnings per tensor. Opt out
  with `options(diffuseR.pin_staging = FALSE)`.
* Video decode runs untiled when the estimated activation cost fits
  the card: in-render decode 12.3 s -> ~1.8 s.
* Attention uses R torch fused scaled-dot-product attention when
  available; the chunked implementation remains the fallback.

## Chained video generation

* `txt2vid_ltx2()` gains `condition_latents=` (an already-encoded
  conditioning prefix, bypassing the VAE), `connector_embeds=`
  (precomputed text-connector outputs — the prompt is constant across
  a chained track, and skipping the per-call connectors phase cuts
  the denoise peak by ~2.5 GiB), `resident=` (keep components on the
  compute device across back-to-back calls, with an idempotent
  onload), and `trim_frames=`. Results carry `latent_shape`;
  `ltx23_tail_latents()` slices a result's trailing latent frames
  into `condition_latents` form.

## Gemma3 text encoder

* `gemma3_quantize_nf4()` + `load_gemma3_nf4()`: the 12B encoder
  quantizes to a ~8 GB NF4 artifact that loads in ~12 s and encodes
  in ~7 s on CUDA (vs ~30 s + ~24 s for fp32 CPU) at 8 GB host RAM
  instead of 45. `load_gemma3_text_encoder()` dispatches to the
  artifact automatically and gains `pin=`: a CPU-resident encoder is
  page-locked once and `encode_with_gemma3()` swaps it to the GPU per
  encode (~0.3 s on, free off).
* `gemma3_encode_batch()`: sub-batched, disk-cached, resumable prompt
  encoding for episode-scale workloads.

## Fixes

* Qwen3 encoder attention masks build in the query dtype, fixing the
  "invalid dtype for bias" error on every FLUX.2 prompt encode through
  fused SDPA.
* `txt2vid_ltx2(decode_audio = FALSE, filename = )` no longer hands
  the raw audio latents to the WAV writer via partial matching.
* Generators accept a three-level `verbose` ("silent", "progress",
  "steps"); "progress" gives a one-line summary plus a progress bar
  or periodic ticks in captured logs.
