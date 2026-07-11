# Parity: yq_flux2_single_block (jitted) vs the diffuseR/torch fixture
# written by tools/gen_fixture_flux2_block.R (real FLUX.2 Klein weights,
# block 0, S = 512). Local-only: needs anvl and the 504MB fixture.

if (!tinytest::at_home()) {
    exit_file("at_home only")
}
if (!requireNamespace("anvl", quietly = TRUE) ||
    !requireNamespace("yunque", quietly = TRUE)) {
    exit_file("anvl not installed")
}
fixture <- file.path(Sys.getenv("HOME"),
                     ".local/share/R/yunque/fixtures/flux2_single_block.safetensors")
if (!file.exists(fixture)) {
    exit_file("fixture missing; run tools/gen_fixture_flux2_block.R")
}

w <- anvl::nv_read(fixture)
expect_equal(anvl::shape(w$w_qkv), c(27648L, 3072L))

w_qkv_t <- anvl::nv_transpose(w$w_qkv)
w_out_t <- anvl::nv_transpose(w$w_out)

blk <- yq_flux2_single_block(heads = 24L, head_dim = 128L)
f <- anvl::jit(blk)

out <- f(w$h, w$shift, w$scale, w$gate, w$cos, w$sin,
         w_qkv_t, w_out_t, w$norm_q_w, w$norm_k_w)
got <- as.array(out)
want <- as.array(w$out)

max_abs <- max(abs(got - want))
mean_abs <- mean(abs(got - want))
cat(sprintf("flux2 single block parity: max %.2e mean %.2e\n",
            max_abs, mean_abs))
expect_true(max_abs < 5e-3)
expect_true(mean_abs < 5e-5)

# jit output must match eager exactly (same graph, same backend)
out_eager <- blk(w$h, w$shift, w$scale, w$gate, w$cos, w$sin,
                 w_qkv_t, w_out_t, w$norm_q_w, w$norm_k_w)
expect_true(max(abs(as.array(out_eager) - got)) < 1e-5)
