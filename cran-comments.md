## Resubmission

This is a resubmission of 0.2.0. The four points raised in review are
addressed:

* Software names are single-quoted in the Description ('Python',
  'Stable Diffusion', 'Hugging Face', with a URL for the last).
* The doubled spaces are gone. They came from trailing whitespace on
  every continuation line, which DCF folding turned into two spaces.
* \value is now present on all 50 exported .Rd files that were missing
  it, describing the class and meaning of the result. (Topic pages that
  document no function, and therefore have no \usage, do not have one.)
* 14 of the 23 \dontrun{} examples were unwrapped and now run during
  check; they were also rewritten to be self-contained rather than
  referring to objects that were never defined. The 10 that remain are
  listed below with the reason for each.

## Test environments

* Ubuntu 24.04 (local), R 4.6.x: R CMD check --as-cran
* Windows 10, R 4.6.0 and R-devel: R CMD check --as-cran, with torch's
  lantern backend installed (so the full suite runs)
* Ubuntu, with lantern deliberately absent, to exercise the path
  win-builder and CRAN take: examples and tests skip gracefully rather
  than erroring
* win-builder, R-devel

## R CMD check results

0 errors | 0 warnings | 1 note

* New submission (checking CRAN incoming feasibility).

The local Linux check additionally shows a NOTE from the torch
package's own load hook in check subprocesses ("could not find
function packageName"); it does not reproduce on any Windows check or
win-builder and is attributable to the local torch build, not this
package.

## Notes for the reviewers

* Ten examples remain under `\dontrun{}`. Each needs multi-GB model
  weights on disk, so none can execute on a check machine:

  - `download_model()`, `download_component()` - network downloads of
    multi-GB weights.
  - `load_pipeline()`, `load_model_component()` - require those
    downloaded weights.
  - `txt2img()`, `txt2img_sd21()`, `txt2img_sdxl()` - full image
    generation; needs the weights and minutes of compute.
  - `ltx23_open_checkpoint()` - needs a ~20 GB LTX-2.3 checkpoint the
    user downloads under Lightricks' own license.
  - `resident_load()` - loads a full pipeline onto a GPU.
  - `load_decoder_weights()` in the `vae_decoder_native()` page -
    needs a checkpoint file. The rest of that example runs.

  `\donttest{}` would not help for these: CRAN runs `\donttest{}`
  examples, and each would either attempt a large download or fail on a
  missing file.

* `save_video()`'s MP4 example is deliberately shown as a comment
  rather than run. Both encoder backends hand off to an ffmpeg process
  that inherits the session's stdin, which `R CMD check --as-cran` uses
  to feed the example script to R, so a live call consumes part of the
  script and later examples parse short. The frame-writing path is
  exercised live instead.

* Model weights are never bundled and never downloaded without
  consent: every download function prompts interactively with the
  size stated, and non-interactive sessions require
  options(diffuseR.consent = TRUE). Downloaded source weights use
  hfhub's user cache; locally derived quantized artifacts are stored
  under tools::R_user_dir("diffuseR", "data").
* torch is in Imports; all tests, examples, and vignette code degrade
  gracefully when torch's backend (lantern) is not installed, as on
  win-builder. Note that `torch::cuda_is_available()` raises an error
  rather than returning FALSE in that state, so the package probes it
  through `tryCatch()` everywhere it is reachable without a GPU.
