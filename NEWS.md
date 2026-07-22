# diffuseR 0.1.0.15 (development)

* Pinned staging now allocates page-locked host memory via
  `torch_empty_strided(pin_memory = TRUE)` instead of the deprecated
  `Tensor$pin_memory(device)` overload, silencing the two libtorch
  deprecation warnings per tensor (thousands of lines per pipeline
  load). Validated at 25.1 GB/s H2D (the card's DMA ceiling) with a
  clean stderr.
* `txt2vid_ltx2()` gains `connector_embeds=`: precomputed
  text-connector outputs (the prompt is constant across a chained
  track) skip the per-call connectors phase. Benchmarked at 960x960
  this cuts the denoise-phase peak by ~2.5 GiB — the per-call phase
  moved the raw Gemma3 hidden-state stack to the GPU, which is what
  OOMed every resident-transformer chain attempt.
* The `resident=` onload check now probes the staging pair's live
  tensor instead of `module$parameters` (which the NF4 transformer
  doesn't expose), so a resident component is never silently
  re-onloaded over itself.
* The Gemma3 loaders gain `pin=` (default: the `diffuseR.pin_staging`
  option): a CPU-resident encoder is page-locked once and
  `encode_with_gemma3()` swaps it to the GPU per encode (~0.3 s on,
  free off) instead of holding VRAM or reloading (~15 s).

# diffuseR 0.1.0.14 (development)

* Latent-space chaining seams for chunked video continuation:
  `txt2vid_ltx2()` gains `condition_latents=` (an already-encoded
  prefix, bypassing the VAE), `resident=` (keep named components on
  the compute device across back-to-back calls; onload is now
  idempotent), and `trim_frames=` (deliver head-free pixels while the
  returned latents keep the full sequence). Results carry
  `latent_shape`, and the new `ltx23_tail_latents()` slices a
  result's trailing latent frames into `condition_latents` form — so
  a chunk chain can stay in latent space end to end: encode once,
  denoise every chunk with the transformer resident, decode once.

# diffuseR 0.1.0.13 (development)

* `gemma3_quantize_nf4()` + `load_gemma3_nf4()` put the Gemma3 12B text
  encoder on the GPU: the 336 projection weights quantize to a ~8 GB
  NF4 artifact (one-time 52 s; vision tower dropped), which loads in
  ~12 s and encodes in ~7 s on CUDA vs ~30 s load + ~24 s encode for
  the fp32 CPU path, at 8 GB host RAM instead of 45. Pinned phase swap
  costs 0.3 s on / 0 s off after a one-time 3.3 s page-lock.
  `load_gemma3_text_encoder()` dispatches to the artifact
  automatically. Embedding drift (cosine 0.989 vs fp32) renders
  scene-equivalent same-seed videos.
* The eager NF4 dequant now shares the byte-LUT with the jit block
  stack (one index per packed byte, gather straight into the output),
  speeding every eager NF4 forward (Gemma3, FLUX.1, LTX fallbacks).
* Fixed `txt2vid_ltx2(decode_audio = FALSE, filename = )`: partial
  matching handed the raw audio latents to the WAV writer.
* `gemma3_encode_batch()` encodes a prompt vector in sub-batches with
  optional per-prompt disk caching (resumable; returns cache paths):
  encode a whole episode list once, then swap the encoder off and
  render from the cached embeddings. Works around a torch_save bug
  where storage-offset views serialize from the base storage.

# diffuseR 0.1.0.12 (development)

* LTX video decode runs untiled when the estimated activation cost
  (~1 GB + ~360 B per output pixel-frame) fits the card - tiling
  bounds VRAM, and at decode time the transformer is phase-offloaded,
  so 768x512x49 decodes in one full-latent forward. The per-tile
  explicit `gc()` is gone (storm-free with the allocator gates live).
  In-render decode phase 12.3 s -> 1.7-1.8 s; warm render 54 -> ~44 s.
  `options(diffuseR.vae_untiled = TRUE/FALSE)` forces either path.

# diffuseR 0.1.0.11 (development)

* `ltx23_tune_gc()` now takes effect: it used to set the allocator gate
  options one call after `torch::cuda_is_available()` had started torch
  (which reads them exactly once at init), so every option it set was
  inert. Options now land before the first torch call, the three CUDA
  gates are pushed into the live allocator, and `.onLoad` defaults
  `torch.threshold_call_gc` (which has no live setter). Measured: ~4-5%
  off LTX render walls (63.7/62.2/61.9 s vs 67.9/64.2/65.6 s at
  768x512x49 NF4).
* `diffuseR.pin_staging` now defaults to TRUE for LTX phase offload:
  page-locked host copies make onload a DMA transfer and offload a
  pointer swap, saving ~7 s per render for +9 s of one-time
  page-locking at pipeline load (break-even on the second render).
  Opt out with `options(diffuseR.pin_staging = FALSE)` under host
  memory pressure.

# diffuseR 0.1.0.10 (development)

* The generators accept a three-level `verbose`: "silent", "progress",
  "steps" (logicals still work: TRUE = "steps", FALSE = "silent").
  "progress" prints a one-line generation summary plus a denoise
  progress bar interactively, or a tick line every ~n/10 steps in
  captured logs, so a long run is distinguishable from a hang without
  per-step noise (#30).

# diffuseR 0.1.0.9 (development)

* The LTX-2.3 JIT block stack dequantizes NF4 weights through a
  precomputed [256, 2] byte lookup table (one embedding gather in the
  compute dtype) instead of the int64 shift/stack/gather chain, cutting
  per-step dequant memory traffic ~6x (isolated benchmark: 5.7x; the
  dequant was the measured per-step wall at ~4.4 s of a 6.5 s step).

# diffuseR 0.1.0.8 (development)

* The Qwen3 encoder builds its additive attention mask in the query
  dtype, fixing the "invalid dtype for bias" CUDA error every FLUX.2
  prompt encode hit through the fused SDPA path (since #33).

# diffuseR 0.1.0.7 (development)

## SDXL native pipeline from safetensors

* `sdxl_pipeline_from_safetensors()` and `txt2img_sdxl(diffusers_dir=)`
  run SDXL end to end from a diffusers safetensors directory (no
  TorchScript, so it works on Blackwell): the two CLIP encoders
  (ViT-L with `quick_gelu`, OpenCLIP bigG) are concatenated at their
  penultimate hidden state into the 2048-dim UNet conditioning, with
  pooled `text_embeds` + `time_ids` added conditioning and the VAE
  `scaling_factor` (0.13025) read from `vae/config.json`.
* `text_encoder_native()` and `text_encoder2_native()` gain
  `return_penultimate` for the SDXL `hidden_states[-2]` prompt embeds;
  the encoder-2 pooled output still comes from the full stack.
* `text_encoder2_native_from_safetensors()` and
  `load_text_encoder2_safetensors()` load the OpenCLIP bigG
  `text_encoder_2` (including its top-level `text_projection`); the
  encoder-1 and encoder-2 loaders now share one key remap and core.
* `reshard_safetensors()` splits an oversize (>= 2 GB) safetensors into
  sub-2 GB diffusers shards plus an index, so large fp16 weights load on
  stock CRAN safetensors.
* `download_sdxl()` fetches the SDXL diffusers weights from the
  `cornball-ai/sdxl-R` dataset.

## LTX-2.3 loading and attention

* Checkpoint loaders build module skeletons (uninitialized weights at
  the target dtype) instead of initializing full-precision modules and
  casting: the LTX-2.3 NF4 pipeline load drops from ~108 GB host RAM
  (kernel OOM on a 125 GB machine) to ~21 GB, and Gemma3 from ~72 GB of
  transient writes to its resident size. `load_gemma3_text_encoder()`
  now errors on any parameter the checkpoint does not fill.
* LTX-2.3 attention now uses R torch fused scaled-dot-product attention
  when available and no explicit query chunk is requested. The adaptive
  chunked/scratch-buffer implementation remains the fallback for older
  torch builds, explicit `attn_chunk`, or `options(diffuseR.ltx23_fused_sdpa = FALSE)`.

# diffuseR 0.1.0.6 (development)

## Uniform native-safetensors + hosted quantization (in progress)

* New `recommend(model, vram_gb, st_caps)`: one VRAM- and
  capability-aware precision/device recommendation for every model
  (sd21, sdxl, flux1, flux2, zimage, ltx). nf4 is the default tier; fp8
  or bf16 is recommended only when the card fits it *and* the installed
  safetensors can **read** that dtype, otherwise it recommends nf4 and
  surfaces the `cornball-ai/safetensors` suggestion in `$note` (never an
  error).
* `flux_memory_profile()` now delegates to `recommend("flux1")`,
  correcting the stale tiers that placed fp8 (GPU-resident now, not
  streamed) in a narrow low-VRAM band it can no longer fit.
* Quantizer shards default to `shard_bytes = 1.9e9` (`flux_quantize`,
  `ltx23_quantize_nf4`, `ltx23_quantize_fp8`). This is *what makes* nf4
  artifacts load on stock CRAN safetensors: R safetensors overflows a
  32-bit offset on any file at or above 2^31 bytes (~2.15 GB), so the
  old 4e9 default produced shards only the fork could read. nf4 is
  CRAN-readable **because of** the sub-2 GB shards, not automatically;
  4e9 remains available for local fork builds.
* Explicitly requesting fp8/bf16 without the needed safetensors support
  now warns and falls back to nf4 instead of failing
  (`download_flux1`, `download_flux2_klein`, `download_zimage_turbo`).
* Reading a legacy oversize (>2 GB) shard on stock safetensors raises an
  actionable "rebuild with smaller shards or install the fork" message
  instead of a raw 32-bit overflow error.
* Native SD21/SDXL UNet weights now load from diffusers safetensors
  (`load_unet_safetensors`, `load_unet_sdxl_safetensors`, and the
  `unet_native_from_safetensors` / `unet_sdxl_native_from_safetensors`
  constructors), with no TorchScript step (Blackwell-safe). The VAE
  decoder and CLIP text encoder already had safetensors loaders; the
  UNet was the gap. Validated against the cached SDXL base UNet (all
  1680 keys map with matching shapes).
* `vae_decoder_native_from_safetensors` and
  `text_encoder_native_from_safetensors` (config-driven CLIP arch
  detection) complete the native SD component set.
* `download_sd21()` + `sd_pipeline_from_safetensors()` run Stable
  Diffusion 2.1 fully natively from diffusers safetensors;
  `txt2img_sd21(diffusers_dir=)` uses it. The SD VAE decode now applies
  the `post_quant_conv` the FLUX-derived native decoder omitted (the
  decode was badly wrong without it). SD 2.1 defaults to float32 on this
  path (fp16 attention overflows to NaN).
* Fixed a native SD 2.1 UNet **tiling** bug (pre-existing; the
  `.pt`-native path shared it). The timestep embedding used
  `flip_sin_to_cos=FALSE, downscale_freq_shift=1`, scrambling the
  sin/cos ordering the trained `time_embedding` weights expect (standard
  diffusers SD, and the native SDXL UNet, use `TRUE/0`). Constant-input
  parity tests could not see it (GroupNorm sits at its bias, attention is
  uniform); it compounded through the spatial path into tiled output.
  The native UNet now matches the TorchScript reference at cos 0.99999,
  and `test_unet.R` gains a random-input parity check.

All of the above is capability-**probed**, not version-pinned, so the
fork requirement self-heals when the safetensors fixes reach CRAN
(mlverse/safetensors#11, #13).
