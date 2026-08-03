# diffuseR 0.2.2

* Every precision `recommend()` can return is now reachable. `bf16` was
  advertised for flux1 at 24 GB and flux2/zimage at 16-24 GB while no
  loader accepted it; the flux-family loaders take
  `precision = "bf16"`, which resolves the unquantized transformer out
  of the hfhub cache rather than an artifact directory (bf16 is the
  source the quantizers read, not something built). `recommend()`
  explains the tier instead of returning a bare string. On a 16 GB
  card, flux2 at bf16 renders 1024x1024 in 6.4 s against 9.1 s at fp8 -
  the highest-quality tier is also the fastest, since nothing
  dequantizes per layer; it costs the 7.8 GB source staying on disk.
* `download_ltx2()` gains `precision = c("nf4", "fp8")` and defaults to
  nf4, which is what `recommend("ltx")` returns for any card with 14 GB
  or more. It previously built only fp8, so the recommended tier had to
  be built by hand with `ltx23_quantize_nf4()`. Asking for fp8 without
  float8 write support now warns and builds nf4 instead of failing
  inside the quantizer.
* Quantized artifacts stay locally built. Prebuilt weights are not
  hosted for any model: only flux2 and zimage could be redistributed
  (Apache-2.0 and ungated), while LTX-2.3 is under the LTX-2 Community
  License and FLUX.1-schnell is gated, so hosting would cover half the
  catalog and leave two models on a different workflow.


* Model residency: `resident_load()`, `resident_activate()`,
  `resident_deactivate()`, `resident_generate()`, `resident_status()`
  and `resident_unload()` keep a pipeline's weights page-locked on the
  host and treat the GPU copy as disposable, so handing a small card
  between models is a DMA transfer rather than a full reload. Same
  contract as whisper and chatterbox, with no `gpu.ctl` dependency.
  This sits above the per-generation phase offloading in the
  `txt2img_*` functions: those swap one component at a time within a
  render, residency decides who owns the card between renders. For a
  phase-offloading pipeline (the default) activation is the ownership
  claim and the transfers stay per-phase; only a pipeline loaded with
  `phase_offload = FALSE` is copied to the card wholesale, and that
  path is checked against free VRAM first. `resident_status()` reports
  `components_on_gpu` alongside `state`, because the two legitimately
  disagree: a render returns every component to pinned host memory as
  its phase ends, so an active handle can hold nothing.

  Verified on an RTX 5060 Ti (16 GB) against local artifacts: flux2
  (11.22 GB pinned, 9.1 s render), flux1 (15.73 GB, 40.1 s), zimage
  (13.45 GB, 19.5 s) and ltx (18.41 GB across 5 components). All three
  image models reproduce bit-for-bit across a deactivate/activate
  cycle. FLUX.1 and LTX both have pinned sets larger than the card, so
  they exercise the refusal path rather than bulk onload.

Addressing the CRAN review of the 0.2.0 submission:

* Every exported `.Rd` with a `\usage` block now documents its return
  value: 50 `@return` tags added, chiefly to the `nn_module` generators
  for the FLUX, FLUX.2, Z-Image, LTX-2.3 and Gemma3 ports.
* Examples: 14 of the 23 `\dontrun{}` blocks now run during check, and
  were rewritten to be self-contained instead of referencing undefined
  objects. The 9 that remain need model weights on disk and are
  itemised in `cran-comments.md`.
* `ddim_scheduler_create()` was uncallable at its documented defaults:
  `beta_schedule` was never passed through `match.arg()`, so `switch()`
  errored on the length-3 default, and the `device` default was a
  length-2 vector that `torch_tensor()` rejects. `ddim_scheduler_step()`
  had the same missing `match.arg()` on `prediction_type`. Every
  internal caller passed these explicitly, so the broken defaults went
  unnoticed. `device` now defaults to `torch_device("cpu")`.
* `DESCRIPTION`: software names single-quoted ('Python', 'Stable
  Diffusion', 'Hugging Face' with its URL) and the trailing whitespace
  that had been folding into double spaces since the first commit
  removed.
* `save_video()`'s mp4 example is no longer live: the encoder inherits
  the session's stdin, which `R CMD check --as-cran` uses to feed the
  example script to R, so it consumed part of the script.

# diffuseR 0.2.0.1

* The FLUX-family image loaders (`flux_load_pipeline`,
  `flux2_load_pipeline`, `zimage_load_pipeline`) now page-lock the
  phase-swapped transformer, VAE decoder, and text encoder(s) at load,
  so the per-generation CPU<->GPU moves run at DMA rate (offload becomes
  a pointer swap). A new `pin` argument, `NULL` by default, resolves via
  `options(diffuseR.pin_staging)` then the host-RAM-aware `recommend()`
  decision. Resident-fp8 transformers (flux2/zimage) stage their fp8
  weight fields too.
* `flux_load_pipeline()` GPU-encodes T5-XXL (bfloat16) on 14 GB+ cards,
  where its encode phase fits; smaller cards keep the float32 CPU
  encode. `text_device` defaults to `NULL` (resolved from the VRAM
  tier). An explicit `text_device = "cpu"` still encodes in place.
* Internal: the pinned-staging helpers lost their `ltx23` prefix
  (`staging.R`); `recommend()` and the loaders share one `.pin_decision`.

# diffuseR 0.2.0

## Serving

* `serve()`: a zero-dependency HTTP server (base R sockets, one
  persistent process, model loaded once) answering OpenAI-style
  requests - `/v1/images/generations` for flux2/zimage/flux1,
  `/v1/videos/generations` for ltx, `GET /health`. Never downloads
  weights; an example systemd unit ships as
  `system.file("diffuser.service", package = "diffuseR")`. Port 7812
  in the cornball serve range. Hardened for
  persistence: optional bearer-token auth, hard pixel/frame limits
  (400 on oversize, including a steps cap, frame-rate bounds, and a
  joint pixels-x-frames video budget), a bounded LRU of per-prompt connector embeds
  (~9 MB each, never the raw Gemma stacks), and a clean process exit
  on CUDA OOM so a supervisor restarts with sane GPU state.
* Every model download is consent-gated: interactive prompt with the
  size stated, and non-interactive sessions require
  `options(diffuseR.consent = TRUE)`. Generation functions never
  download implicitly.
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
