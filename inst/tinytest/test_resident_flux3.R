# resident_load("flux3") dispatches to flux3_load_pipeline and
# txt2img_flux3; the loader reads its text encoder, VAE and tokenizer
# from given paths (text_dir, support_dir), and pinned_bytes counts the
# sets FLUX 3 streams off pipeline$staging. No GPU or weights needed.

library(tinytest)
library(diffuseR)

expect_true("flux3" %in% diffuseR:::.resident_families)
expect_true("flux3" %in% eval(formals(resident_load)$model))
expect_true(all(c("text_dir", "support_dir") %in%
                names(formals(flux3_load_pipeline))))
expect_error(resident_load("flux3", text_encoder = tempdir()),
             "ltx family only")

# text_dir precision comes from the directory itself
prec <- diffuseR:::.flux3_text_dir_precision
root <- tempfile("flux3-text-")
fp8 <- file.path(root, "fp8")
dir.create(fp8, recursive = TRUE)
writeLines('{"model": "qwen3", "format": "fp8", "shards": ["a.safetensors"]}',
           file.path(fp8, "manifest.json"))
file.create(file.path(fp8, "a.safetensors"))
expect_equal(prec(fp8), "fp8")
bf16 <- file.path(root, "bf16")
dir.create(bf16)
writeLines("{}", file.path(bf16, "config.json"))
file.create(file.path(bf16, "model-00001-of-00002.safetensors"))
expect_equal(prec(bf16), "bf16")
empty <- file.path(root, "empty")
dir.create(empty)
expect_error(prec(empty), "Not a Qwen3-VL text encoder")
unlink(root, recursive = TRUE)

# pinned_bytes: streamed Qwen3-VL layers plus the fp8 linears past
# resident_gb
if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
    exit_file("torch not fully installed")
}
pair <- function(n) {
    list(live = NULL, pinned = torch::torch_zeros(n, dtype = torch::torch_bfloat16()))
}
te <- torch::nn_linear(2, 2)
te$layer_stream <- list(list(pair(10)), list(pair(6), pair(4)))
lin <- function(n) {
    m <- torch::nn_linear(2, 2)
    m$weight_fp8 <- torch::torch_zeros(n, dtype = torch::torch_uint8())
    m
}
pipe <- list(text_encoder = te,
             fp8_split = list(resident = list(lin(100)),
                              streamed = list(lin(7), lin(5))))
expect_equal(diffuseR:::.resident_flux3_stream_bytes(pipe), 2 * 20 + 12)
expect_equal(diffuseR:::.resident_flux3_stream_bytes(
    list(text_encoder = torch::nn_linear(2, 2))), 0)
