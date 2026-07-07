# Pinned staging round trip (R/staging_ltx23.R): pin -> onload ->
# offload -> onload must preserve outputs exactly. CUDA-only.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}
if (!torch::cuda_is_available()) exit_file("no CUDA")

library(diffuseR)
torch::torch_manual_seed(13)

m <- ltx23_feed_forward(16L)
m$eval()
x <- torch::torch_randn(2L, 5L, 16L)
torch::with_no_grad(ref <- m(x))

st <- diffuseR:::.ltx23_pin_component(m)
expect_false(is.null(st))
expect_true(suppressWarnings(
  st[[1]]$live$is_pinned(device = torch::torch_device("cuda"))
))

# Pinning must not change CPU outputs
torch::with_no_grad(out_pinned <- m(x))
expect_true(as.numeric((out_pinned - ref)$abs()$max()) == 0)

# Onload: GPU forward matches
diffuseR:::.ltx23_staged_onload(st, "cuda")
expect_equal(st[[1]]$live$device$type, "cuda")
torch::with_no_grad(
  out_gpu <- m(x$to(device = "cuda"))$cpu()
)
expect_true(as.numeric((out_gpu - ref)$abs()$max()) < 1e-5)

# Offload: pointer swap back to the pinned copies, exact outputs
diffuseR:::.ltx23_staged_offload(st)
expect_equal(st[[1]]$live$device$type, "cpu")
torch::with_no_grad(out_back <- m(x))
expect_true(as.numeric((out_back - ref)$abs()$max()) == 0)

# Second round trip still exact
diffuseR:::.ltx23_staged_onload(st, "cuda")
torch::with_no_grad(out_gpu2 <- m(x$to(device = "cuda"))$cpu())
expect_true(as.numeric((out_gpu2 - out_gpu)$abs()$max()) == 0)
diffuseR:::.ltx23_staged_offload(st)
