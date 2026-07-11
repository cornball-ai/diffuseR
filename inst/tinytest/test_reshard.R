# reshard_safetensors: split a single safetensors into sub-2 GB shards
# with a diffusers index, and read it back through the sharded reader.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}
if (!requireNamespace("safetensors", quietly = TRUE)) {
  exit_file("safetensors not installed")
}
library(diffuseR)

# three 100x100 float32 tensors (~40 KB each)
set.seed(1)
tensors <- list(a = torch::torch_randn(100L, 100L),
                b = torch::torch_randn(100L, 100L),
                c = torch::torch_randn(100L, 100L))
src <- tempfile(fileext = ".safetensors")
safetensors::safe_save_file(tensors, src)

out <- tempfile()
# ~60 KB target forces one tensor per shard -> three shards
reshard_safetensors(src, out, base = "model", shard_bytes = 60000,
  verbose = FALSE)

idx <- file.path(out, "model.safetensors.index.json")
expect_true(file.exists(idx))
j <- jsonlite::fromJSON(idx, simplifyVector = FALSE)
expect_equal(length(j$weight_map), 3L)                 # every key mapped
expect_true(length(unique(unlist(j$weight_map))) >= 2) # actually sharded
# diffusers -of- naming
expect_true(all(grepl("model-\\d{5}-of-\\d{5}\\.safetensors",
  unlist(j$weight_map))))

# read back through the sharded reader: same keys, same values
opened <- diffuseR:::.flux_open_sharded_dir(out, "model")
expect_true(setequal(opened$keys, c("a", "b", "c")))
for (k in c("a", "b", "c")) {
  expect_true(as.logical(torch::torch_allclose(
    opened$handle$get_tensor(k), tensors[[k]])))
}

unlink(c(src, out), recursive = TRUE)
