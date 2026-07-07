# FLUX.2 quantization round trip on the tiny sharded checkpoint:
# family auto-detection via config _class_name, cast census, NF4 and
# fp8 (streamed + resident) loads. Everything runs on CPU.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}
if (!requireNamespace("safetensors", quietly = TRUE)) {
  exit_file("safetensors not installed")
}

library(diffuseR)

ckpt_dir <- system.file("tinytest", "fixtures", "flux2_tiny_ckpt",
  package = "diffuseR")
if (ckpt_dir == "") ckpt_dir <- "fixtures/flux2_tiny_ckpt"
if (!dir.exists(ckpt_dir)) exit_file("flux2 tiny checkpoint missing")

fixture_path <- system.file("tinytest", "fixtures", "flux2_model.safetensors",
  package = "diffuseR")
if (fixture_path == "") fixture_path <- "fixtures/flux2_model.safetensors"
if (!file.exists(fixture_path)) exit_file("flux2 model fixtures missing")

fx <- safetensors::safe_load_file(fixture_path, framework = "torch")

max_abs_diff <- function(a, b) {
  as.numeric(torch::torch_max(torch::torch_abs(
    a$to(dtype = torch::torch_float32()) - b$to(dtype = torch::torch_float32())
  )))
}
cosine_sim <- function(a, b) {
  a <- a$to(dtype = torch::torch_float32())$flatten()
  b <- b$to(dtype = torch::torch_float32())$flatten()
  as.numeric(torch::torch_dot(a, b) / (a$norm() * b$norm()))
}

# --- family detection + cast census ------------------------------------------------
# Tiny config: 1 double x 12 + 1 single x 2 + 3 modulations +
# context_embedder = 18 cast weights. (Full klein-4B: 104.)

ckpt <- flux_open_checkpoint(ckpt_dir)
expect_equal(diffuseR:::.flux_family(ckpt$config), "flux2")
expect_equal(sum(flux2_is_quant_key(ckpt$keys)), 18L)

# --- full-precision load through family dispatch ------------------------------------

model <- flux_load_transformer(ckpt, device = "cpu", dtype = "float32",
  verbose = FALSE)
ids <- torch::torch_cat(list(fx$txt_ids, fx$img_ids), dim = 1L)
rope <- flux_pos_embed(ids, axes_dim = c(2L, 2L, 2L, 2L), theta = 2000)
out_full <- torch::with_no_grad(model(
  hidden_states = fx$hidden,
  encoder_hidden_states = fx$encoder,
  timestep = fx$timestep,
  image_rotary_emb = rope
))
expect_true(max_abs_diff(out_full, fx$out) < 1e-4)

# --- NF4 round trip ------------------------------------------------------------------

nf4_dir <- file.path(tempdir(), "flux2-tiny-nf4")
unlink(nf4_dir, recursive = TRUE)
manifest <- flux_quantize(ckpt_dir, output_dir = nf4_dir, format = "nf4",
  verbose = FALSE)
expect_equal(manifest$cast, 18L)
expect_true(grepl("^flux2-klein-nf4", manifest$shards[[1]]))

model_nf4 <- flux_load_transformer(flux_open_quantized(nf4_dir),
  device = "cpu", verbose = FALSE)
out_nf4 <- torch::with_no_grad(model_nf4(
  hidden_states = fx$hidden$to(dtype = torch::torch_bfloat16()),
  encoder_hidden_states = fx$encoder$to(dtype = torch::torch_bfloat16()),
  timestep = fx$timestep,
  image_rotary_emb = rope
))
expect_true(cosine_sim(out_nf4, out_full) > 0.98)
ltx23_release_dequant_buffers()

# --- fp8 round trips (streamed and resident) ------------------------------------------

f8_ok <- tryCatch({
  x <- torch::torch_randn(2, 2)$to(dtype = torch::torch_float8_e4m3fn())
  tmp <- tempfile(fileext = ".safetensors")
  safetensors::safe_save_file(list(w = x), tmp)
  y <- safetensors::safe_load_file(tmp, framework = "torch")
  unlink(tmp)
  TRUE
}, error = function(e) FALSE)

if (f8_ok) {
  fp8_dir <- file.path(tempdir(), "flux2-tiny-fp8")
  unlink(fp8_dir, recursive = TRUE)
  manifest8 <- flux_quantize(ckpt_dir, output_dir = fp8_dir, format = "fp8",
    verbose = FALSE)
  expect_equal(manifest8$cast, 18L)

  model_fp8 <- flux_load_transformer(flux_open_quantized(fp8_dir),
    device = "cpu", pin = FALSE, verbose = FALSE)
  out_fp8 <- torch::with_no_grad(model_fp8(
    hidden_states = fx$hidden$to(dtype = torch::torch_bfloat16()),
    encoder_hidden_states = fx$encoder$to(dtype = torch::torch_bfloat16()),
    timestep = fx$timestep,
    image_rotary_emb = rope
  ))
  expect_true(cosine_sim(out_fp8, out_full) > 0.99)

  # Resident variant: weights moved to the compute device (CPU here just
  # exercises the walker), forward unchanged
  model_res <- flux_load_transformer(flux_open_quantized(fp8_dir),
    device = "cpu", pin = FALSE, fp8_resident = TRUE, verbose = FALSE)
  out_res <- torch::with_no_grad(model_res(
    hidden_states = fx$hidden$to(dtype = torch::torch_bfloat16()),
    encoder_hidden_states = fx$encoder$to(dtype = torch::torch_bfloat16()),
    timestep = fx$timestep,
    image_rotary_emb = rope
  ))
  expect_true(max_abs_diff(out_res, out_fp8) == 0)
}

options(diffuseR.block_gc = NULL)
