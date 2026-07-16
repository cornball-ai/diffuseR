# Skeleton construction (noinit_modules.R): linear_noinit /
# embedding_noinit initialize exactly like their torch counterparts on
# bare construction, skip initialization inside .construct_skeleton(),
# and the skeleton scope sets/restores the torch default dtype.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}

library(diffuseR)

skel <- diffuseR:::.construct_skeleton
linear_noinit <- diffuseR:::linear_noinit
embedding_noinit <- diffuseR:::embedding_noinit
noinit_state <- diffuseR:::.noinit_state

# --- bare construction initializes like torch -------------------------------------

torch::torch_manual_seed(7)
l <- linear_noinit(8L, 4L)
expect_true(all(is.finite(as.numeric(l$weight))))
expect_true(all(is.finite(as.numeric(l$bias))))
# kaiming_uniform(a = sqrt(5)) bound and the bias bound are both 1/sqrt(fan_in)
expect_true(as.numeric(l$weight$abs()$max()) <= 1 / sqrt(8) + 1e-6)
expect_true(as.numeric(l$bias$abs()$max()) <= 1 / sqrt(8) + 1e-6)

l_nobias <- linear_noinit(8L, 4L, bias = FALSE)
expect_null(l_nobias$bias)

e <- embedding_noinit(16L, 4L)
expect_true(all(is.finite(as.numeric(e$weight))))

# --- forward parity with torch modules --------------------------------------------

ref <- torch::nn_linear(8L, 4L)
torch::with_no_grad({
  ref$weight$copy_(l$weight)
  ref$bias$copy_(l$bias)
})
x <- torch::torch_randn(2L, 8L)
expect_true(as.numeric((l(x) - ref(x))$abs()$max()) == 0)

eref <- torch::nn_embedding(16L, 4L)
torch::with_no_grad(eref$weight$copy_(e$weight))
idx <- torch::torch_randint(1L, 16L, size = c(2L, 3L), dtype = torch::torch_long())
expect_true(as.numeric((e(idx) - eref(idx))$abs()$max()) == 0)

# --- skeleton scope: target dtype, state restored ----------------------------------

old_dtype <- torch::torch_get_default_dtype()
m <- skel(linear_noinit, 8L, 4L, dtype = torch::torch_float16())
expect_true(m$weight$dtype == torch::torch_float16())
expect_true(m$bias$dtype == torch::torch_float16())
expect_true(torch::torch_get_default_dtype() == old_dtype)
expect_false(noinit_state$active)

me <- skel(embedding_noinit, 16L, 4L, dtype = torch::torch_bfloat16())
expect_true(me$weight$dtype == torch::torch_bfloat16())

# state restored even when the constructor throws
expect_error(skel(function() stop("boom")), "boom")
expect_false(noinit_state$active)
expect_true(torch::torch_get_default_dtype() == old_dtype)

# --- skeleton weights are usable once filled --------------------------------------

m2 <- skel(linear_noinit, 4L, 2L, dtype = torch::torch_float32())
torch::with_no_grad({
  m2$weight$fill_(0.5)
  m2$bias$fill_(0)
})
y <- m2(torch::torch_ones(1L, 4L))
expect_true(as.numeric((y - 2)$abs()$max()) < 1e-6)
