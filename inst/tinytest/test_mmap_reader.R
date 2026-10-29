# .mmap_reader: tensors read through a private memory map of a
# safetensors file match safetensors' own reader exactly, for every
# dtype the artifacts use, including tensors whose data is not aligned
# to their element size (the R writer does not pad the header).

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed() ||
    !requireNamespace("safetensors", quietly = TRUE)) {
  exit_file("torch or safetensors not available")
}
library(torch)

torch_manual_seed(3)
src <- list(
  a_u8_odd = torch_randint(0, 255, 7L, dtype = torch_uint8()),
  b_bf16 = torch_randn(3, 5)$to(dtype = torch_bfloat16()),
  c_f32 = torch_randn(4, 2, 3),
  d_u8 = torch_randint(0, 255, c(2L, 3L), dtype = torch_uint8()),
  e_i64 = torch_tensor(c(-3, 0, 9), dtype = torch_long()),
  f_scalar = torch_tensor(2.5),
  g_f16 = torch_randn(5)$to(dtype = torch_float16())
)
f1 <- tempfile(fileext = ".safetensors")
f2 <- tempfile(fileext = ".safetensors")
# Lengthen one key until the header length leaves typed data misaligned
# (the case the reader exists for)
misaligned <- function(f) {
  h <- diffuseR:::.st_header(f)
  el <- c(U8 = 1, BF16 = 2, F32 = 4, I64 = 8, F16 = 2)
  any(vapply(h$tensors, function(i) {
    (h$data_start + i$data_offsets[[1]]) %% el[[i$dtype]] != 0
  }, logical(1)))
}
for (pad in 0:7) {
  names(src)[1] <- paste0("a_u8_odd", strrep("x", pad))
  safetensors::safe_save_file(src[1:4], f1)
  if (misaligned(f1)) break
}
expect_true(misaligned(f1))
safetensors::safe_save_file(src[5:7], f2)

reader <- diffuseR:::.mmap_reader(c(f1, f2))
if (is.null(reader)) {
  exit_file("this torch cannot map files")
}
expect_equal(sort(reader$keys()), sort(names(src)))

for (k in names(src)) {
  got <- reader$read(k, "cpu")
  expect_equal(got$dtype, src[[k]]$dtype, info = k)
  expect_equal(as.integer(got$shape), as.integer(src[[k]]$shape), info = k)
  expect_true(torch_equal(got, src[[k]]), info = k)
}
expect_error(reader$read("nope", "cpu"), "not found")

# A write to a read tensor reaches neither the file nor later reads
x <- reader$read("d_u8", "cpu")
x$add_(1L)
fresh <- safetensors::safetensors$new(f1, framework = "torch")$get_tensor("d_u8")
expect_true(torch_equal(fresh, src$d_u8))
expect_true(torch_equal(reader$read("d_u8", "cpu"), src$d_u8))

if (torch::cuda_is_available()) {
  for (k in names(src)) {
    got <- reader$read(k, "cuda")
    expect_equal(got$device$type, "cuda", info = k)
    expect_true(torch_equal(got$cpu(), src[[k]]), info = k)
  }
}

# Disk-backed staging through the reader: placeholders after offload,
# exact values after onload
live <- lapply(src[c("b_bf16", "c_f32")], function(t) t$clone())
pairs <- diffuseR:::.stage_from_disk(live, c(b_bf16 = "b_bf16", c_f32 = "c_f32"),
                                     list(get_tensor = function(k) stop("unused"),
                                          read_to = reader$read), pin = FALSE)
expect_equal(live$c_f32$stride(), c(0L, 0L, 0L))
diffuseR:::.staged_onload(pairs, "cpu")
expect_true(torch_equal(live$c_f32, src$c_f32))
expect_true(torch_equal(live$b_bf16, src$b_bf16))
diffuseR:::.staged_offload(pairs)
expect_equal(live$b_bf16$stride(), c(0L, 0L))

unlink(c(f1, f2))
