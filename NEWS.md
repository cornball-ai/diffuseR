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
