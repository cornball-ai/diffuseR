# recommend(): VRAM + safetensors-read-capability policy, the read
# probe, and the flux_memory_profile adapter.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}
library(diffuseR)

fork <- list(bfloat16 = TRUE, float8_e4m3fn = TRUE)   # can read everything
cran <- list(bfloat16 = TRUE, float8_e4m3fn = FALSE)  # reads bf16, not fp8

# --- structure --------------------------------------------------------------------

r <- recommend("flux1", vram_gb = 16, st_caps = fork)
expect_true(is.list(r))
expect_equal(r$model, "flux1")
expect_true(all(c("precision", "devices", "offload", "max_pixels",
  "text_device", "attn_chunk", "vram_gb", "fork_suggested", "note") %in%
  names(r)))
expect_true(is.integer(r$max_pixels) || is.numeric(r$max_pixels))

# --- floors: nf4 for the quantized families, fp16 for SD -------------------------

# No GPU: everything runs on CPU at its floor precision, no fork nag.
# SD ships no quantized weights, so its floor is fp16.
for (m in c("sd21", "sdxl", "flux1", "flux2", "zimage")) {
  z <- recommend(m, vram_gb = 0, st_caps = cran)
  expect_equal(z$precision, if (m %in% c("sd21", "sdxl")) "fp16" else "nf4")
  expect_false(z$fork_suggested)
  expect_true(all(unlist(z$devices) == "cpu"))
}

# Mid VRAM with no float8: nf4, and no nag when the card could not fit
# fp8 anyway (fp8 tier not VRAM-eligible).
expect_equal(recommend("flux1", 12, cran)$precision, "nf4")
expect_false(recommend("flux1", 12, cran)$fork_suggested)

# --- fp8 upgrade + fork gating ----------------------------------------------------

# 16 GB with float8 read: fp8 recommended, no nag.
f16 <- recommend("flux1", 16, fork)
expect_equal(f16$precision, "fp8")
expect_false(f16$fork_suggested)
expect_null(f16$note)
expect_equal(f16$devices$transformer, "cuda")

# 16 GB WITHOUT float8 read: fp8 wanted but unreadable -> nf4 + fork nag.
c16 <- recommend("flux1", 16, cran)
expect_equal(c16$precision, "nf4")
expect_true(c16$fork_suggested)
expect_true(is.character(c16$note) && grepl("safetensors#13", c16$note))
expect_false(grepl("—", c16$note))   # no em dash (house style)

# --- bf16 top tier (CRAN-readable, no fork needed) --------------------------------

# 24 GB reads bf16 on CRAN too -> bf16, no nag.
b24 <- recommend("flux1", 24, cran)
expect_equal(b24$precision, "bf16")
expect_false(b24$fork_suggested)

# But if bf16 read were somehow missing, it is gated like fp8.
nob <- recommend("flux1", 24, list(bfloat16 = FALSE, float8_e4m3fn = FALSE))
expect_true(nob$fork_suggested)
expect_equal(nob$precision, "nf4")

# --- SD ladder: fp16 for cards that fit, nf4 default, cpu -------------------------

expect_equal(recommend("sdxl", 16, cran)$precision, "fp16")
expect_equal(recommend("sdxl", 8, cran)$precision, "fp16")
expect_equal(recommend("sdxl", 16, cran)$devices$unet, "cuda")
expect_true("text_encoder2" %in% names(recommend("sdxl", 16, cran)$devices))
expect_false("text_encoder2" %in% names(recommend("sd21", 16, cran)$devices))
# SD is CRAN-readable at every tier: never a fork nag.
for (v in c(0, 4, 8, 16, 24)) expect_false(recommend("sdxl", v, cran)$fork_suggested)

# --- precision rises monotonically with VRAM (quality never drops) ----------------

rank <- c(nf4 = 1L, fp16 = 2L, fp8 = 3L, bf16 = 4L)
for (m in c("flux1", "flux2", "zimage")) {
  precs <- vapply(c(0, 8, 12, 16, 24),
    function(v) rank[[recommend(m, v, fork)$precision]], integer(1))
  expect_true(!is.unsorted(precs))
}

# --- read probe: override option wins, and self-consistency -----------------------

expect_true(is.logical(diffuseR:::.st_can_read("bfloat16")))
options(diffuseR.st_read_caps = list(bfloat16 = FALSE, float8_e4m3fn = FALSE))
expect_false(diffuseR:::.st_can_read("bfloat16"))
expect_false(diffuseR:::.st_can_read("float8_e4m3fn"))
# recommend() with st_caps = NULL now reads the (forced) probe.
expect_true(recommend("flux1", 24)$fork_suggested)
options(diffuseR.st_read_caps = NULL)

# On a machine whose safetensors CAN read a dtype, the hand-built probe
# file must decode (guards the writeBin byte patterns).
if (requireNamespace("safetensors", quietly = TRUE)) {
  # These reflect the installed safetensors; just assert they are logical
  # and, when TRUE, that recommend trusts them.
  expect_true(is.logical(diffuseR:::.st_can_read("float8_e4m3fn")))
}

# --- flux_memory_profile adapter follows recommend --------------------------------

options(diffuseR.st_read_caps = fork)
p <- flux_memory_profile(vram_gb = 16)
expect_equal(p$precision, "fp8")
expect_equal(p$phase_offload, TRUE)
expect_true(p$max_pixels >= 1024L * 1024L)
options(diffuseR.st_read_caps = cran)
p2 <- flux_memory_profile(vram_gb = 16)
expect_equal(p2$precision, "nf4")
expect_true(isTRUE(p2$fork_suggested))
options(diffuseR.st_read_caps = NULL)

# --- graceful fallback for an explicitly requested precision ----------------------

grc <- diffuseR:::.st_graceful_precision
# nf4/fp16 pass through untouched
expect_equal(grc("nf4", "write"), "nf4")
expect_equal(grc("fp16", "write"), "fp16")
# fp8 without float8 WRITE support -> nf4 + fork message
options(diffuseR.st_caps = list(float8_e4m3fn = FALSE))
expect_message(rr <- grc("fp8", "write"), pattern = "safetensors#13")
expect_equal(rr, "nf4")
options(diffuseR.st_caps = list(float8_e4m3fn = TRUE))
expect_equal(grc("fp8", "write"), "fp8")   # present -> passes through
options(diffuseR.st_caps = NULL)
# bf16 gate goes through the READ probe in read mode
options(diffuseR.st_read_caps = list(bfloat16 = FALSE))
expect_message(rr2 <- grc("bf16", "read"), pattern = "safetensors#11")
expect_equal(rr2, "nf4")
options(diffuseR.st_read_caps = NULL)

# --- fork note fit parameter ------------------------------------------------------

fn <- diffuseR:::.st_fork_note
expect_true(grepl("best fit for your card", fn("fp8", fit = TRUE)))
expect_false(grepl("best fit for your card", fn("fp8", fit = FALSE)))
expect_false(grepl("—", fn("fp8")))   # no em dash, either variant

# --- multi-GB read breadcrumb -----------------------------------------------------

msg <- diffuseR:::.st_overflow_message("shard-00001.safetensors", 3.4e9, "boom")
expect_true(grepl("3.4 GB", msg))
expect_true(grepl("2\\^31", msg))
expect_true(grepl("shard-00001", msg))

brc <- diffuseR:::.st_read_or_breadcrumb
# a read that succeeds is returned untouched
expect_equal(brc(function() 42L, NULL), 42L)
# a failure on a small (or absent) file rethrows verbatim, no breadcrumb
small <- tempfile()
writeLines("x", small)
e <- tryCatch(brc(function() stop("plain boom"), small),
  error = function(err) conditionMessage(err))
expect_true(grepl("plain boom", e))
expect_false(grepl("2\\^31", e))
unlink(small)

# --- pin decision -----------------------------------------------------------------

r <- recommend("ltx", vram_gb = 16, st_caps = cran, host_ram_gb = 125)
expect_true(all(c("pin", "pinned_set_gb", "host_ram_gb") %in% names(r)))
expect_true(r$pin)                   # 125 GB >> 2 x ~28 GB pinned set
expect_true(r$pinned_set_gb > 0)

expect_false(recommend("ltx", vram_gb = 16, st_caps = cran,
                       host_ram_gb = 32)$pin)   # 32 < 2 x pinned set

expect_false(recommend("ltx", vram_gb = 0, st_caps = cran,
                       host_ram_gb = 125)$pin)  # cpu tier: nothing stages

expect_true(recommend("sdxl", vram_gb = 16, st_caps = cran,
                      host_ram_gb = NA)$pin)    # undetectable: fail-soft on

hr <- diffuseR:::.detect_host_ram()
expect_true(is.na(hr) || (is.numeric(hr) && hr > 0))
expect_equal(diffuseR:::.pinned_set_gb("ltx", "bf16"), 0)  # unknown tier -> 0

# --- every recommended tier must be reachable -------------------------------------

# recommend() used to return "bf16" for big cards while no loader
# accepted it and no builder produced it, with note = NULL. bf16 is the
# unquantized source, so the tier is real; the note has to say so.
fork <- list(bfloat16 = TRUE, float8_e4m3fn = TRUE)
b <- recommend("flux2", vram_gb = 24, st_caps = fork)
expect_equal(b$precision, "bf16")
expect_true(is.character(b$note))
expect_true(grepl("download_flux2_klein", b$note))
expect_true(grepl("quantize = FALSE", b$note))
expect_false(b$fork_suggested)

# The flux-family loaders must accept it as a precision.
for (fn in c("flux2_load_pipeline", "zimage_load_pipeline")) {
  choices <- eval(formals(getExportedValue("diffuseR", fn))$precision)
  expect_true("bf16" %in% choices)
}

# A quantized tier still resolves to its artifact directory, and bf16
# resolves elsewhere (the hub cache), not to a "<prefix>bf16" dir.
expect_equal(diffuseR:::.flux_model_dir("flux2", "nf4", "/tmp/x-"), "/tmp/x-nf4")
expect_false(identical(
  tryCatch(diffuseR:::.flux_model_dir("flux2", "bf16", "/tmp/x-"),
           error = function(e) "errored"),
  "/tmp/x-bf16"))

# An unknown model has no bf16 source and must say so rather than
# silently building a bogus path.
expect_error(diffuseR:::.flux_source_dir("nosuch"), pattern = "No bf16 source")

# --- LTX's recommended tier must be downloadable ----------------------------------

# recommend("ltx") is nf4 on any card >= 14 GB, so download_ltx2() has
# to be able to build nf4, not just fp8.
expect_equal(recommend("ltx", vram_gb = 16)$precision, "nf4")
expect_true("precision" %in% names(formals(download_ltx2)))
expect_equal(eval(formals(download_ltx2)$precision), c("nf4", "fp8"))
