# SD/SDXL UNet safetensors loader: HF-key remap rules, the copy path,
# completeness, and error reporting. The remap is validated against real
# cached SDXL weights out of band (all 1680 keys map with matching
# shapes); here we exercise the loader logic portably through a mock
# module so no multi-GB checkpoint is needed.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}
if (!requireNamespace("safetensors", quietly = TRUE)) {
  exit_file("safetensors not installed")
}
library(diffuseR)

r21 <- diffuseR:::.unet_remap_sd21
rxl <- diffuseR:::.unet_remap_sdxl

# --- remap rules ------------------------------------------------------------------

expect_equal(r21("time_embedding.linear_1.weight"),
  "time_embedding_linear_1.weight")
expect_equal(r21("time_embedding.linear_2.bias"),
  "time_embedding_linear_2.bias")
# non-embedding keys pass through untouched (dotted block paths)
expect_equal(r21("down_blocks.0.resnets.0.norm1.weight"),
  "down_blocks.0.resnets.0.norm1.weight")
expect_equal(r21("conv_out.weight"), "conv_out.weight")
# SDXL adds add_embedding and keeps the time_embedding rule
expect_equal(rxl("add_embedding.linear_1.weight"),
  "add_embedding_linear_1.weight")
expect_equal(rxl("add_embedding.linear_2.bias"), "add_embedding_linear_2.bias")
expect_equal(rxl("time_embedding.linear_1.weight"),
  "time_embedding_linear_1.weight")
# SD21 rule leaves add_embedding alone (SD21 has none)
expect_equal(r21("add_embedding.linear_1.weight"),
  "add_embedding.linear_1.weight")

# --- round-trip through a mock module ---------------------------------------------

native_names <- c("conv_in.weight", "conv_in.bias",
  "time_embedding_linear_1.weight", "time_embedding_linear_1.bias",
  "time_embedding_linear_2.weight",
  "down_blocks.0.resnets.0.norm1.weight", "add_embedding_linear_1.weight")

# inverse remap (native -> HF) to author a synthetic checkpoint
to_hf <- function(n) {
  n <- sub("^time_embedding_linear_1", "time_embedding.linear_1", n)
  n <- sub("^time_embedding_linear_2", "time_embedding.linear_2", n)
  sub("^add_embedding_linear_1", "add_embedding.linear_1", n)
}

make_params <- function(gen) {
  p <- lapply(native_names, function(i) gen())
  names(p) <- native_names
  p
}
truth <- make_params(function() torch::torch_randn(2L, 3L))

dir <- tempfile()
dir.create(dir)
hf <- truth
names(hf) <- vapply(native_names, to_hf, character(1))
safetensors::safe_save_file(hf,
  file.path(dir, "diffusion_pytorch_model.safetensors"))

# fresh zeroed params; the loader must copy truth into them
params <- make_params(function() torch::torch_zeros(2L, 3L))
mock <- list(named_parameters = function() params)
diffuseR:::.load_unet_safetensors(mock, dir, rxl, "mock UNet", verbose = FALSE)
for (nm in native_names) {
  expect_true(as.logical(torch::torch_allclose(params[[nm]], truth[[nm]])))
}

# passing the file directly (not the dir) also works
params_b <- make_params(function() torch::torch_zeros(2L, 3L))
mock_b <- list(named_parameters = function() params_b)
diffuseR:::.load_unet_safetensors(mock_b,
  file.path(dir, "diffusion_pytorch_model.safetensors"), rxl, "mock",
  verbose = FALSE)
expect_true(as.logical(torch::torch_allclose(params_b[["conv_in.weight"]],
  truth[["conv_in.weight"]])))

# --- error paths ------------------------------------------------------------------

# extra HF key with no native destination -> unmapped
dir2 <- tempfile()
dir.create(dir2)
hf_extra <- hf
hf_extra[["nonexistent.layer.weight"]] <- torch::torch_zeros(2L, 3L)
safetensors::safe_save_file(hf_extra,
  file.path(dir2, "diffusion_pytorch_model.safetensors"))
expect_error(
  diffuseR:::.load_unet_safetensors(
    list(named_parameters = function() make_params(function() torch::torch_zeros(2L, 3L))),
    dir2, rxl, "mock", verbose = FALSE),
  pattern = "unmapped")

# a native param with no checkpoint key -> unfilled
params_missing <- c(make_params(function() torch::torch_zeros(2L, 3L)),
  list("extra.param.weight" = torch::torch_zeros(2L, 3L)))
expect_error(
  diffuseR:::.load_unet_safetensors(
    list(named_parameters = function() params_missing), dir, rxl, "mock",
    verbose = FALSE),
  pattern = "unfilled")

# shape mismatch -> error (checked before unfilled)
params_wrong <- make_params(function() torch::torch_zeros(2L, 3L))
params_wrong[["conv_in.weight"]] <- torch::torch_zeros(5L, 5L)
expect_error(
  diffuseR:::.load_unet_safetensors(
    list(named_parameters = function() params_wrong), dir, rxl, "mock",
    verbose = FALSE),
  pattern = "shape mismatch")

unlink(c(dir, dir2), recursive = TRUE)

# --- end-to-end against real SDXL weights (opt-in; heavy ~9.6 GB load) -------------
# Runs only with DIFFUSER_TEST_SDXL_LOAD=1 and the cached diffusers UNet
# present, so a normal suite run never pulls 9.6 GB. Validated manually:
# builds + loads all 1680 params in ~20 s.
sdxl_dir <- Sys.glob(file.path("~/.cache/huggingface/hub",
  "models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/*/unet"))
if (at_home() && nzchar(Sys.getenv("DIFFUSER_TEST_SDXL_LOAD")) &&
    length(sdxl_dir) && dir.exists(sdxl_dir[1])) {
  m <- unet_sdxl_native_from_safetensors(sdxl_dir[1], verbose = FALSE)
  expect_equal(length(m$named_parameters()), 1680L)
  w <- m$named_parameters()[["conv_in.weight"]]
  expect_true(as.numeric(w$abs()$sum()$item()) > 0)
  expect_false(m$training)
  rm(m)
  gc()
}
