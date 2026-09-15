# diffuseR 0.2.2.10

* **A pinned component whose onload failed partway no longer stays
  wedged.** The Gemma3 encoder's staged encode, and the LTX pipeline's
  per-phase onload, decided "already on the card" by probing the FIRST
  staging pair. An onload that dies partway -- device memory runs out
  with most of the encoder copied -- leaves exactly that pair on the card
  and the rest on the host, so every later call skipped the onload and
  failed on the first matrix multiply with "mat2 is on cpu", on every
  request, until the process ended. That is what took the gpuhost's
  ltx-2.3 entry down for USA 20260912 on 2026-09-10. Three changes:
  `.staged_on()` asks every pair, not the first; `.staged_onload()` is
  idempotent per pair, so a resident component is a no-op (no re-transfer
  over itself) and a partial one is completed; and `encode_with_gemma3()`
  arms its offload BEFORE the onload, so a failed transfer is undone on
  the way out and the next encode starts from a clean host copy. Covered
  by `test_staged_on.R` (pure fakes, no GPU) and a partial round trip in
  `test_staging.R` (CUDA).

* **Staging compares the card, not just the device type.** A request
  for `"cuda:1"` no longer counts a tensor on `cuda:0` as resident, so a
  multi-GPU caller asking for a particular card gets its weights moved
  there instead of a skipped transfer and a device mismatch. A request
  for bare `"cuda"` still accepts any card, as before.

* **`resident_unload()` drops the LTX text encoder.** The encoder a
  resident LTX handle loads (0.2.2.8) sits outside `staging` by design,
  and unload never released it: an unloaded handle kept the encoder's
  pinned buffers while reporting `pinned_bytes = 0`. Both review findings
  from the 2026-09-11 Codex pass.

# diffuseR 0.2.2.9

* **A resident LTX encoder now stages the prompt encode to the card
  instead of running it on CPU.** `txt2vid_ltx2()` chose the encode device
  with `if (is.character(text_encoder)) device else "cpu"` -- so a
  PRELOADED encoder (the resident/gpuhost path, `resident_load("ltx",
  text_encoder = ...)`) always encoded on CPU, even though the resident
  loader page-locks it with `pin = TRUE` for exactly the staged transfer
  `encode_with_gemma3()` supports. The pinned staging sat unused and every
  prompt paid the ~24 s CPU encode instead of the ~7 s staged-GPU one; on
  the gpuhost path that is once per chunk. The device decision is now
  `.ltx23_text_encode_device()`: a path loads onto the asked-for device, a
  preloaded encoder with a `staging` set and a cuda request stages to the
  card, and a bare CPU-resident object still degrades to CPU (a cuda
  request without staging would be a device mismatch). Pure and unit-tested
  without a GPU (`test_text_encode_device.R`).

# diffuseR 0.2.2.8

* `flux2_load_pipeline()` takes a `revision`. Its VAE, Qwen3 encoder and
  tokenizer come from the Hugging Face cache, and hfhub's default revision
  is the branch `main` -- resolved through `refs/main` and, failing that,
  over the network. A read-only bind of one snapshot carries neither, so
  the load failed there; an exact 40-hex commit takes hfhub straight to
  `snapshots/<revision>/<file>`. A branch name is refused rather than
  passed through.

* `resident_load("ltx", ...)` takes `text_encoder` and `tokenizer` paths.
  `ltx23_load_pipeline()` does not load them and `txt2vid_ltx2()` takes
  them per call, so a resident LTX handle could be activated and could not
  generate -- and a serving caller passing paths re-read 7.6 GB of Gemma3
  on every request. Given here they load once, pinned on the host, and
  `resident_generate()` supplies them. They stay OUT of the handle's
  staging on purpose: `resident_activate()` places everything in staging at
  once, and the encoder does not fit beside the transformer. It rides to
  the card for its own phase and back off, as the pipeline's components do.
  `pinned_bytes` counts it.

# diffuseR 0.2.2.7

* `recommend()` diagnosed the wrong safetensors capability for bf16. The
  tier gate consults `.st_can_read()`, but the note it produced cited
  mlverse/safetensors#11, which is the bfloat16 *write* fix. bfloat16
  read worked on CRAN 0.2.1, so a reader that lacks it is not waiting on
  #11, and the message sent users to an unrelated issue. `.st_update_note()`
  now takes `mode`, and both call sites pass the capability they actually
  gated on. float8 is unchanged: 0.2.1 had neither read nor write, so #13
  is correct from both sides.

  The suite had been holding this in place. Its read-mode assertion
  matched `"safetensors#11"`, so the wrong reference was pinned by the
  test rather than caught by it. The remedy-only guards added earlier in
  this release could not catch it either: the message named the right
  package to install while blaming the wrong missing feature.

* safetensors 0.3.0 reached CRAN on 2026-08-21 with all four fixes
  diffuseR had been routing users around: float8 support
  (mlverse/safetensors#13), bfloat16 write (#11), the >2 GB offset
  overflow (#14), and empty tensor names (#10). Every message that told
  users to install a development build from GitHub now tells them to run
  `install.packages("safetensors")`. That covers `recommend()`'s tier
  note, the graceful fp8/bf16 fallback, both `flux_quantize()` errors, and
  the >2 GB read breadcrumb.

  The same sweep reached the documentation, which is where most of the
  stale advice actually was: the `shard_bytes` help for `flux_quantize()`,
  `ltx23_quantize_nf4()`, `ltx23_quantize_fp8()` and
  `gemma3_quantize_nf4()` described the 1.9e9 default as what "stock CRAN
  safetensors" can read and pointed at a fork for anything larger.
  `README.md`, `vignette("performance-levers")`, `reshard_safetensors()`
  and the `unet_safetensors`, `download_prebuilt` and
  `convert_sd21_pt_to_diffusers` help pages carried variants of the same.

  All of it now names the capability rather than a version: "needs the
  overflow fix (mlverse/safetensors#14, which reached CRAN in 0.3.0)"
  rather than "requires safetensors 0.3.0 or newer". That distinction is
  the package's whole stance here, documented below: a version number
  never separated a capable build from an incapable one, so prose making
  0.3.0 the requirement contradicts the probes it sits beside. The >2 GB
  read breadcrumb and the bf16 resident-dtype message use the same
  vocabulary.

  The capability probes are unchanged, and deliberately so. They were
  written as runtime probes rather than a version floor precisely so this
  day would need no code change, and they still cover what a version test
  cannot: the fixes existed for three weeks in builds that reported 0.2.1.
  No version floor has been added to `Suggests` for the same reason: nf4
  works on older safetensors, so a stale install costs a tier rather than
  the model.

  `recommend()`'s returned `fork_suggested` field keeps its name, which is
  now historical: it means the installed safetensors cannot read a tier
  the card could otherwise run. Renaming it would break the returned
  contract for a cosmetic gain.

* `reshard_safetensors()` is no longer required to make a large artifact
  readable, since 0.3.0 fixed the overflow it worked around. It stays
  useful for publishing: the shards it writes load on every safetensors
  including the older ones, which is what makes a hosted artifact safe to
  redistribute.

# diffuseR 0.2.2.6

* Fixed an allocator pre-warm accumulation introduced in 0.2.2.4.
  `.resident_prewarm()` requested the full onload need on every
  activation, which doubled the CUDA caching allocator's pool after each
  render: a generation fragments the cache, so the next single large
  request cannot be served from it and takes a fresh allocation beside the
  old one. Under `resident_deactivate(release = FALSE)` nothing empties
  the cache, so SDXL went 5.299 GiB after one cycle to 10.322 after two
  and refused the third. It now measures the free cache on the handle's
  own device and grows only the shortfall, skipping entirely when the pool
  already covers the transfer; a cold pool is unchanged, and the
  cold-start win is intact (2.48 s against 2.51 s before). Measured flat
  at 5.396 / 5.398 / 5.398 / 5.398 / 5.398 GiB across five cycles, with a
  phase-offloading family (flux2) untouched because it never takes the
  bulk branch.

  This also corrected the budget independently of the refusal: both
  release modes doubled between the first and second cycle, so any peak
  measured on a single activation understated steady state roughly 2x.

* `resident_generate()`'s documented return value was wrong. It claimed
  `flux1`, `flux2` and `zimage` return bare image arrays and `sdxl` was
  the exception. Every family returns a list: the five image families
  return `list(image, metadata)`, so `$image` unwraps uniformly across all
  five. `ltx` returns `latents`, `audio_latents`, `latent_shape` and
  `sample_rate`, plus `video` and `audio` when `decode_video` /
  `decode_audio` are TRUE — a caller that turns either off gets a list
  without that field rather than a NULL one. Only visibility differs:
  `txt2img_sdxl()` and `txt2img_sd21()` use `return()`, the rest
  `invisible()`.

# diffuseR 0.2.2.5

* `resident_load()` accepts `"sd21"`, the sixth resident family.
  `sd21_load_pipeline()` defaults to the `download_sd21()` cache and pins
  the UNet at **float32**: SD 2.1's attention overflows in float16 and the
  pipeline returns all-NaN rather than raising, so a float16 resident would
  generate blank images with nothing to catch it. Only the UNet is placed
  on the card, matching what `auto_devices("sd21")` already recommends.

* `txt2img_sd21()` no longer requires the legacy TorchScript `.pt` files
  when it is handed a pipeline it did not build, matching the same fix made
  for `txt2img_sdxl()` in 0.2.2.4.

* Known limitation, unchanged by this release but now measured: SD 2.1's
  native float32 path does not fit its own 768x768 default on a 15.47 GiB
  card. The denoise wants 11.35 GB with a further 3.1 GB of allocator
  slack, with or without residency (a plain `txt2img_sd21()` OOMs at
  11.494 GB). 512x512 uses 6.807 GB and is comfortable.

# diffuseR 0.2.2.4

* `resident_load()` accepts `"sdxl"`, making it the fifth resident family.
  `sdxl_load_pipeline()` is the new adapter: it defaults to the
  `download_sdxl()` cache, fixes the UNet at float16 before the weights are
  pinned, and marks the pipeline so that only the UNet is placed on the
  card. Onloading all four components fits the 8.0 GB of weights and then
  OOMs in the VAE decode, which runs 1024x1024 in float32 while the UNet is
  still resident; the text encode and decode therefore run on the host from
  the same pinned copies. `resident_generate()` supplies the matching
  `devices` so `txt2img_sdxl()` does not re-decide the placement with
  `auto_devices()`, and an explicit `devices` from the caller still wins.

* `resident_deactivate()` now releases the NF4 dequantisation buffers for
  `ltx`. They live in a package-level environment rather than in the module,
  so offloading the weights did not free them and `gc()` could not reclaim
  them, leaving scratch on the card after every deactivation.

* A bulk activation pre-warms the CUDA caching allocator. Growing the pool
  one allocation per tensor dominated the first activation: 24.16 s against
  0.32 s once warm.

* `txt2img_sdxl()` no longer requires the legacy TorchScript `.pt` files
  when it is handed a pipeline it did not build, and `setup_dtype()` accepts
  an ordinal-qualified device such as `"cuda:0"`.

# diffuseR 0.2.2.3

* Prebuilt NF4 artifacts for flux2 (2.1 GB) and zimage (3.5 GB) are now
  hosted on the cornball-ai HuggingFace org, and
  `download_flux2_klein()` / `download_zimage_turbo()` fetch them by
  default when the resolved precision is nf4 (`prebuilt = FALSE` forces
  a local build). This narrowly reverses 0.2.2's no-hosting decision:
  the CRAN safetensors cannot read any of the multi-GB upstream sources
  (the fix is merged upstream, mlverse/safetensors#14, but unreleased),
  which left a stock CRAN install unable to build a quantized artifact
  at all. Hosting the two redistributable models (both Apache-2.0,
  ungated) gives a plain `install.packages("diffuseR")` setup something
  to generate with right away. FLUX.1-schnell (gated repo) and LTX-2.3
  (LTX-2 Community License) still download sources and build locally.

# diffuseR 0.2.2.2

* `recommend()` tier selection now applies a 0.5 GB tolerance to the
  `min_vram` thresholds (#56). The thresholds are nameplate card sizes,
  but detection reports free VRAM and no card reports its nameplate as
  free, so every nameplate-valued tier (the flux-family 8 GB tiers, the
  SDXL 12 GB tier, the flux2 bf16 16 GB tier) was unreachable on
  exactly the card it targets and silently dropped to the CPU tier.

# diffuseR 0.2.2.1

* `txt2img()` and `img2img()` now `match.arg()` their `model_name`, so
  the bare calls work (defaulting to sd21) instead of erroring on the
  choices vector.
* README fixes for the CRAN-rendered page: closed the unclosed fence
  that swallowed the LTX section, replaced the LTX example with the
  working call shape, and pointed the example images at GitHub URLs
  (the files are .Rbuildignore'd).

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
  objects. The 10 that remain need model weights on disk and are
  itemised in `cran-comments.md` (the nine left from that pass, plus
  `resident_load()`, added below).
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
