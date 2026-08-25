# safetensors capability gating: dtype write probes, precision "auto"
# resolution, and the quantizer's resident-dtype fallback for the CRAN
# safetensors 0.2.1 release (no bfloat16 write, no float8).

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}
if (!requireNamespace("safetensors", quietly = TRUE)) {
  exit_file("safetensors not installed")
}

library(diffuseR)

can_write <- diffuseR:::.st_can_write
resolve_precision <- diffuseR:::.flux_resolve_precision

# --- capability probe -------------------------------------------------------------
expect_cran_remedy <- function(msg) {
  expect_true(grepl('install.packages("safetensors")', msg, fixed = TRUE))
  expect_false(grepl("development version", msg, fixed = TRUE))
  expect_false(grepl("GitHub", msg, fixed = TRUE))
  invisible(msg)
}

expect_true(is.logical(can_write("bfloat16")))
expect_true(is.logical(can_write("float8_e4m3fn")))

# Option override wins over the probe (and its cache)
options(diffuseR.st_caps = list(bfloat16 = FALSE, float8_e4m3fn = FALSE))
expect_false(can_write("bfloat16"))
expect_false(can_write("float8_e4m3fn"))
options(diffuseR.st_caps = list(float8_e4m3fn = TRUE))
expect_true(can_write("float8_e4m3fn"))
options(diffuseR.st_caps = NULL)

# --- precision "auto" resolution ---------------------------------------------------

# Explicit precision passes through untouched
expect_equal(resolve_precision("nf4"), "nf4")
expect_equal(resolve_precision("fp8"), "fp8")

# An existing artifact wins, fp8 preferred. Pin the fp8 capability so
# the assertion is hermetic: on CRAN safetensors 0.2.1 the ambient probe
# is FALSE and resolve_precision correctly refuses the fp8 artifact.
options(diffuseR.st_caps = list(float8_e4m3fn = TRUE))
prefix <- file.path(tempdir(), "stcaps-test-")
nf4_dir <- paste0(prefix, "nf4")
fp8_dir <- paste0(prefix, "fp8")
dir.create(nf4_dir, showWarnings = FALSE)
writeLines("{}", file.path(nf4_dir, "manifest.json"))
expect_equal(resolve_precision("auto", prefix), "nf4")
dir.create(fp8_dir, showWarnings = FALSE)
writeLines("{}", file.path(fp8_dir, "manifest.json"))
expect_equal(resolve_precision("auto", prefix), "fp8")
unlink(c(nf4_dir, fp8_dir), recursive = TRUE)
options(diffuseR.st_caps = NULL)

# No artifact: capability decides
options(diffuseR.st_caps = list(float8_e4m3fn = FALSE))
expect_equal(resolve_precision("auto", prefix), "nf4")
options(diffuseR.st_caps = list(float8_e4m3fn = TRUE))
expect_equal(resolve_precision("auto", prefix), "fp8")
options(diffuseR.st_caps = NULL)

# An fp8 artifact present but unreadable (CRAN safetensors 0.2.1) is NOT
# selected - it would fail at read time. Only nf4 is safe there.
dir.create(fp8_dir, showWarnings = FALSE)
writeLines("{}", file.path(fp8_dir, "manifest.json"))
options(diffuseR.st_caps = list(float8_e4m3fn = FALSE))
expect_equal(resolve_precision("auto", prefix), "nf4")
# With float8 support the same fp8 artifact IS selected
options(diffuseR.st_caps = list(float8_e4m3fn = TRUE))
expect_equal(resolve_precision("auto", prefix), "fp8")
options(diffuseR.st_caps = NULL)
unlink(fp8_dir, recursive = TRUE)

# --- quantizer gates (tiny checkpoint, CRAN 0.2.1 emulation) -------------------------

ckpt_dir <- system.file("tinytest", "fixtures", "zimage_tiny_ckpt",
  package = "diffuseR")
if (ckpt_dir == "") ckpt_dir <- "fixtures/zimage_tiny_ckpt"
if (!dir.exists(ckpt_dir)) exit_file("zimage tiny checkpoint missing")

options(diffuseR.st_caps = list(bfloat16 = FALSE, float8_e4m3fn = FALSE))

# fp8 quantization is refused with an actionable error
fp8_err <- tryCatch(
  flux_quantize(ckpt_dir, file.path(tempdir(), "stcaps-fp8"),
    format = "fp8", verbose = FALSE),
  error = function(e) conditionMessage(e)
)
expect_true(is.character(fp8_err))
expect_true(grepl("float8", fp8_err, fixed = TRUE))
expect_cran_remedy(fp8_err)

# The load-path gate has its own hand-written error and must carry the
# same remedy even when this installation cannot build a real fp8 artifact.
# The fixture must carry the class: flux_load_transformer() checks
# inherits(ckpt, "ltx23_checkpoint") BEFORE the fp8 gate, so a bare list
# errors on the stopifnot and never reaches the message under test.
load_ckpt <- structure(list(format = "fp8"), class = "ltx23_checkpoint")
load_err <- tryCatch(
  flux_load_transformer(load_ckpt, device = "cpu", verbose = FALSE),
  error = function(e) conditionMessage(e)
)
expect_true(is.character(load_err))
expect_true(grepl("float8", load_err, fixed = TRUE))
expect_cran_remedy(load_err)

# NF4 quantization falls back to float32 residents
nf4_out <- file.path(tempdir(), "stcaps-nf4")
unlink(nf4_out, recursive = TRUE)
expect_message(
  manifest <- flux_quantize(ckpt_dir, nf4_out, format = "nf4",
    verbose = FALSE),
  pattern = "float32"
)
ck <- flux_open_quantized(nf4_out)
resident <- ck$handle$get_tensor("cap_embedder.1.weight")
expect_equal(as.character(resident$dtype), "Float")

# And the artifact still loads and runs
model <- flux_load_transformer(ck, device = "cpu", dtype = "float32",
  verbose = FALSE)
out <- torch::with_no_grad(model(
  torch::torch_randn(4L, 1L, 12L, 20L),
  torch::torch_tensor(0.5)$reshape(1L),
  torch::torch_randn(37L, 24L)
))
expect_equal(out$shape, c(4L, 1L, 12L, 20L))
ltx23_release_dequant_buffers()

options(diffuseR.st_caps = NULL)
options(diffuseR.block_gc = NULL)
unlink(c(nf4_out, file.path(tempdir(), "stcaps-fp8")), recursive = TRUE)

# --- load-path guard: fp8 artifact opened without float8 support --------------------
# Build a real fp8 artifact (needs actual float8 support), then force the
# probe to "no float8" and confirm loading errors actionably rather than
# with a raw dtype failure.
if (can_write("float8_e4m3fn")) {
  fp8_out <- file.path(tempdir(), "stcaps-loadguard-fp8")
  unlink(fp8_out, recursive = TRUE)
  flux_quantize(ckpt_dir, fp8_out, format = "fp8", verbose = FALSE)
  options(diffuseR.st_caps = list(float8_e4m3fn = FALSE))
  expect_error(
    flux_load_transformer(flux_open_quantized(fp8_out), device = "cpu",
      verbose = FALSE),
    pattern = "float8"
  )
  options(diffuseR.st_caps = NULL)
  unlink(fp8_out, recursive = TRUE)
}
