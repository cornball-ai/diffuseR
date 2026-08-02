## Test environments

* Ubuntu 24.04 (local), R 4.6.x: R CMD check --as-cran
* Windows 10, R 4.6.0 (full test suite against installed torch backend)
* Windows 10, R-devel (2026-07-21 r90286), torch installed without its
  lantern backend (tests skip gracefully)
* win-builder, R-devel (2026-07-22 r90289)

## R CMD check results

0 errors | 0 warnings | 1 note

* New submission (checking CRAN incoming feasibility).

The local Linux check additionally shows a NOTE from the torch
package's own load hook in check subprocesses ("could not find
function packageName"); it does not reproduce on any Windows check or
win-builder and is attributable to the local torch build, not this
package.

## Notes for the reviewers

* On the previous submission we were asked to replace `\dontrun{}`
  with `\donttest{}`. We unwrapped 14 of the 23 examples so they now
  run during check (device/profile policy helpers, both schedulers,
  the VRAM helpers, `save_image()`, `save_video()` frame output, and a
  small `vae_decoder_native()`), and rewrote them to be self-contained
  rather than referencing undefined objects. Nine remain under
  `\dontrun{}` because they cannot execute anywhere without model
  weights on disk:

  - `download_model()`, `download_component()` - network downloads of
    multi-GB weights.
  - `load_pipeline()`, `load_model_component()` - require those
    downloaded weights.
  - `txt2img()`, `txt2img_sd21()`, `txt2img_sdxl()` - full image
    generation; needs the weights and minutes of compute.
  - `ltx23_open_checkpoint()` - needs a ~20 GB LTX-2.3 checkpoint the
    user downloads under Lightricks' own license.
  - `load_decoder_weights()` in the `vae_decoder_native()` page -
    needs a checkpoint file. The rest of that example runs.

  `\donttest{}` would not help for these: CRAN runs `\donttest{}`
  examples, and each of these would either attempt a large download or
  fail on a missing file.

* Model weights are never bundled and never downloaded without
  consent: every download function prompts interactively with the
  size stated, and non-interactive sessions require
  options(diffuseR.consent = TRUE). Downloaded source weights use
  hfhub's user cache; locally derived quantized artifacts are stored
  under tools::R_user_dir("diffuseR", "data").
* torch is in Imports; all tests, examples, and vignette code degrade
  gracefully when torch's backend (lantern) is not installed, as on
  win-builder.
