# The bf16 tier reads the unquantized HuggingFace source instead of a
# built artifact, so the risk it carries is cache layout: does
# .flux_source_dir() land on a directory the loader can actually open,
# and is what it finds really bf16?
#
# Scope, deliberately: this checks resolution, on-disk dtype, and that
# flux_open_checkpoint() reports the "full" format that routes to the
# bf16 load path. It does NOT run a forward pass. bf16 names the storage
# dtype, and reading it is CPU work, but *computing* in bf16 is GPU-only
# here -- txt2img_flux2() upcasts to float32 when device == "cpu" -- and
# materializing the 7.8 GB transformer is not a test-suite job.
#
# at_home() gates the whole file: R CMD check machines have no such
# cache.

library(tinytest)
library(diffuseR)

if (!at_home()) {
  exit_file("bf16 source checks need a populated hfhub cache")
}
if (!requireNamespace("safetensors", quietly = TRUE)) {
  exit_file("safetensors not installed")
}

src <- tryCatch(diffuseR:::.flux_source_dir("flux2"), error = function(e) NULL)
if (is.null(src)) {
  exit_file("FLUX.2 bf16 source not in the hfhub cache")
}

# --- resolution lands somewhere the loader can use --------------------------------

expect_true(dir.exists(src))
expect_true(file.exists(file.path(src, "config.json")))
expect_true(length(list.files(src, pattern = "[.]safetensors$")) > 0L)

# The precision router must send bf16 here and NOT to "<prefix>bf16",
# which is what it did before and why the tier was unreachable.
prefix <- file.path(tools::R_user_dir("diffuseR", "data"), "flux2-klein-4b-")
expect_equal(diffuseR:::.flux_model_dir("flux2", "bf16", prefix), src)
expect_equal(diffuseR:::.flux_model_dir("flux2", "nf4", prefix),
             paste0(prefix, "nf4"))

# --- what is on disk is actually bf16 ---------------------------------------------

# Read only the safetensors JSON header (8-byte little-endian length,
# then that many bytes), so this stays cheap and loads no tensors.
shard <- list.files(src, pattern = "[.]safetensors$", full.names = TRUE)[1]
con <- file(shard, "rb")
n <- readBin(con, "integer", n = 1L, size = 8L, endian = "little")
hdr <- rawToChar(readBin(con, "raw", n = n))
close(con)
meta <- jsonlite::fromJSON(hdr)
meta <- meta[names(meta) != "__metadata__"]
dtypes <- unique(vapply(meta, function(x) x$dtype, character(1)))
expect_equal(dtypes, "BF16")

# --- the checkpoint opens and reports the full (unquantized) format ---------------

ckpt <- flux_open_checkpoint(src)
expect_inherits(ckpt, "ltx23_checkpoint")
# flux_load_transformer() branches on `format %||% "full"`; a source
# checkpoint carries no format, which is what selects the bf16 path.
expect_null(ckpt$format)
expect_true(length(ckpt$keys) > 0L)
expect_equal(length(ckpt$keys), length(meta))
expect_equal(ckpt$config[["_class_name"]], "Flux2Transformer2DModel")
