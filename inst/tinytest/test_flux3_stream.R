# FLUX 3 small-card streaming: the VRAM plan, Qwen3-VL layer streaming
# and the fp8 resident/streamed split. On the CPU the staged moves are
# host-to-host, so outputs must match the unstreamed model exactly.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}
if (!requireNamespace("safetensors", quietly = TRUE)) {
  exit_file("safetensors not installed")
}

library(diffuseR)

max_abs_diff <- function(a, b) {
  as.numeric(torch::torch_max(torch::torch_abs(
    a$to(dtype = torch::torch_float32()) - b$to(dtype = torch::torch_float32())
  )))
}

# --- Plan ---------------------------------------------------------------------

plan <- diffuseR:::.flux3_stream_plan
expect_equal(plan(16), list(stream_text = FALSE, resident_gb = Inf))
expect_equal(plan(12), list(stream_text = FALSE, resident_gb = Inf))
expect_equal(plan(8), list(stream_text = TRUE, resident_gb = 1.5))
expect_equal(plan(10)$stream_text, FALSE)
expect_equal(plan(10)$resident_gb, 3.5)
expect_equal(plan(6)$resident_gb, 0)
# No GPU detected: nothing streams
expect_equal(plan(0), list(stream_text = FALSE, resident_gb = Inf))

# --- Qwen3-VL layer streaming -----------------------------------------------------

ckpt_dir <- system.file("tinytest", "fixtures", "flux3_qwen_tiny_ckpt",
  package = "diffuseR")
if (ckpt_dir == "") ckpt_dir <- "fixtures/flux3_qwen_tiny_ckpt"
if (dir.exists(ckpt_dir)) {
  enc <- load_qwen3_text_encoder(ckpt_dir, device = "cpu", dtype = "float32",
    verbose = FALSE)
  ids <- torch::torch_tensor(matrix(c(5L, 9L, 13L, 2L, 7L), nrow = 1),
    dtype = torch::torch_long())
  ref <- torch::with_no_grad(enc(ids, out_layers = c(1L, 2L, 4L)))

  base <- diffuseR:::.flux3_text_stream(enc, pin = FALSE)
  # One staging set per layer; embeddings + final norm in the base set
  expect_equal(length(enc$layer_stream), length(enc$model$layers))
  expect_equal(length(base), 2L)
  n_layer_tensors <- sum(vapply(enc$layer_stream, length, 0L))
  expect_equal(n_layer_tensors,
    length(enc$parameters) - 2L)
  diffuseR:::.staged_onload(base, "cpu")
  out <- torch::with_no_grad(enc(ids, out_layers = c(1L, 2L, 4L)))
  for (k in seq_along(ref)) {
    expect_true(max_abs_diff(out[[k]], ref[[k]]) == 0)
  }
  enc$layer_stream <- NULL
}

# --- fp8 split -----------------------------------------------------------------------

tiny_dir <- system.file("tinytest", "fixtures", "flux3_tiny_ckpt",
  package = "diffuseR")
if (tiny_dir == "") tiny_dir <- "fixtures/flux3_tiny_ckpt"
fixture_path <- system.file("tinytest", "fixtures", "flux3_model.safetensors",
  package = "diffuseR")
if (fixture_path == "") fixture_path <- "fixtures/flux3_model.safetensors"
f8_ok <- diffuseR:::.st_can_write("float8_e4m3fn")
if (f8_ok && dir.exists(tiny_dir) && file.exists(fixture_path)) {
  fx <- safetensors::safe_load_file(fixture_path, framework = "torch")
  fp8_dir <- file.path(tempdir(), "flux3-tiny-fp8-stream")
  unlink(fp8_dir, recursive = TRUE)
  manifest <- flux_quantize(tiny_dir, output_dir = fp8_dir, format = "fp8",
    verbose = FALSE)
  model <- flux_load_transformer(flux_open_quantized(fp8_dir), device = "cpu",
    dtype = "float32", pin = FALSE, verbose = FALSE)
  run <- function() {
    torch::with_no_grad(model(hidden_states = fx$hidden,
      encoder_hidden_states = fx$encoder, timestep = fx$timestep,
      img_ids = fx$img_ids, txt_ids = fx$txt_ids,
      image_cond_hidden_states = fx$cond, image_cond_ids = fx$cond_ids,
      target_timestep = fx$target_timestep))
  }
  ref <- run()

  all <- diffuseR:::.flux3_fp8_split(model, Inf, pin = FALSE)
  expect_equal(length(all$resident), manifest$cast)
  expect_equal(length(all$streamed), 0L)
  none <- diffuseR:::.flux3_fp8_split(model, 0, pin = FALSE)
  expect_equal(length(none$streamed), manifest$cast)

  half <- all$resident_gb / 2
  split <- diffuseR:::.flux3_fp8_split(model, half, pin = FALSE)
  expect_true(split$resident_gb <= half)
  expect_true(length(split$resident) > 0L && length(split$streamed) > 0L)
  expect_equal(length(split$resident) + length(split$streamed), manifest$cast)
  expect_equal(length(diffuseR:::.flux3_fp8_fields(split$resident)),
    2L * length(split$resident))
  expect_true(max_abs_diff(run(), ref) == 0)
  ltx23_release_dequant_buffers()
  unlink(fp8_dir, recursive = TRUE)
}
options(diffuseR.block_gc = NULL)
