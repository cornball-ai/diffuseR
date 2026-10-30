# .ltx23_video_array copies the decoded clip to an R array a few frames
# at a time; the result must match the whole-clip conversion exactly,
# for any chunk size and with leading frames trimmed

library(tinytest)

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
    exit_file("torch not fully installed")
}

torch::torch_manual_seed(1)
v <- torch::torch_randn(1, 3, 9, 6, 10)$mul(0.8)$to(dtype = torch::torch_bfloat16())
whole <- as.array(((v$to(dtype = torch::torch_float32()) / 2 + 0.5)$
    clamp(0, 1))[1, , , , ]$permute(c(2L, 3L, 4L, 1L))$cpu())

for (ch in c(1L, 4L, 9L, 16L)) {
    expect_identical(diffuseR:::.ltx23_video_array(v, chunk = ch), whole)
    expect_identical(diffuseR:::.ltx23_video_array(v, skip = 3L, chunk = ch),
                     whole[-(1:3), , , , drop = FALSE])
}
expect_identical(dim(diffuseR:::.ltx23_video_array(v, skip = 8L)), c(1L, 6L, 10L, 3L))
