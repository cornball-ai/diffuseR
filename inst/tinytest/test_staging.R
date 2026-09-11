# Pinned staging round trip (R/staging.R): pin -> onload ->
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

st <- diffuseR:::.pin_component(m)
expect_false(is.null(st))
expect_true(suppressWarnings(
  st[[1]]$live$is_pinned(device = torch::torch_device("cuda"))
))

# Pinning must not change CPU outputs
torch::with_no_grad(out_pinned <- m(x))
expect_true(as.numeric((out_pinned - ref)$abs()$max()) == 0)

# Onload: GPU forward matches
diffuseR:::.staged_onload(st, "cuda")
expect_equal(st[[1]]$live$device$type, "cuda")
torch::with_no_grad(
  out_gpu <- m(x$to(device = "cuda"))$cpu()
)
expect_true(as.numeric((out_gpu - ref)$abs()$max()) < 1e-5)

# Offload: pointer swap back to the pinned copies, exact outputs
diffuseR:::.staged_offload(st)
expect_equal(st[[1]]$live$device$type, "cpu")
torch::with_no_grad(out_back <- m(x))
expect_true(as.numeric((out_back - ref)$abs()$max()) == 0)

# Second round trip still exact
diffuseR:::.staged_onload(st, "cuda")
torch::with_no_grad(out_gpu2 <- m(x$to(device = "cuda"))$cpu())
expect_true(as.numeric((out_gpu2 - out_gpu)$abs()$max()) == 0)
diffuseR:::.staged_offload(st)

# PARTIAL onload, the state a failed transfer leaves behind: the first
# pair on the card, the rest on the host. The whole-set check must call
# it not resident, and the next onload must complete it -- moving only
# what is missing -- rather than skip it on the strength of pair 1.
st[[1]]$live$set_data(st[[1]]$pinned$to(device = "cuda"))
expect_equal(st[[1]]$live$device$type, "cuda")
expect_equal(st[[length(st)]]$live$device$type, "cpu")
expect_false(diffuseR:::.staged_on(st, "cuda"))
diffuseR:::.staged_onload(st, "cuda")
expect_true(diffuseR:::.staged_on(st, "cuda"))
torch::with_no_grad(out_gpu3 <- m(x$to(device = "cuda"))$cpu())
expect_true(as.numeric((out_gpu3 - out_gpu)$abs()$max()) == 0)
diffuseR:::.staged_offload(st)
expect_true(diffuseR:::.staged_on(st, "cpu"))
torch::with_no_grad(out_back2 <- m(x))
expect_true(as.numeric((out_back2 - ref)$abs()$max()) == 0)
