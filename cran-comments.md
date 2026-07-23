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

* Model weights are never bundled and never downloaded without
  consent: every download function prompts interactively with the
  size stated, and non-interactive sessions require
  options(diffuseR.consent = TRUE). Weights land under
  tools::R_user_dir("diffuseR", "data").
* torch is in Imports; all tests, examples, and vignette code degrade
  gracefully when torch's backend (lantern) is not installed, as on
  win-builder.
