# Parity: yq_flux2_double_block (jitted) vs the diffuseR/torch fixture
# written by tools/gen_fixture_flux2_double.R (real FLUX.2 Klein
# transformer_blocks.0 weights, S_txt=64, S_img=256). Local-only.

if (!tinytest::at_home()) {
    exit_file("at_home only")
}
if (!requireNamespace("anvl", quietly = TRUE) ||
    !requireNamespace("yunque", quietly = TRUE)) {
    exit_file("anvl not installed")
}
fixture <- file.path(Sys.getenv("HOME"),
                     ".local/share/R/yunque/fixtures/flux2_double_block.safetensors")
if (!file.exists(fixture)) {
    exit_file("fixture missing; run tools/gen_fixture_flux2_double.R")
}

f <- anvl::nv_read(fixture)
tt <- function(x) anvl::nv_transpose(x)   # [out, in] -> [in, out]

w <- list(
    to_q = tt(f[["attn.to_q.weight"]]),
    to_k = tt(f[["attn.to_k.weight"]]),
    to_v = tt(f[["attn.to_v.weight"]]),
    norm_q = f[["attn.norm_q.weight"]],
    norm_k = f[["attn.norm_k.weight"]],
    add_q_proj = tt(f[["attn.add_q_proj.weight"]]),
    add_k_proj = tt(f[["attn.add_k_proj.weight"]]),
    add_v_proj = tt(f[["attn.add_v_proj.weight"]]),
    norm_added_q = f[["attn.norm_added_q.weight"]],
    norm_added_k = f[["attn.norm_added_k.weight"]],
    to_out = tt(f[["attn.to_out.0.weight"]]),
    to_add_out = tt(f[["attn.to_add_out.weight"]]),
    ff_in = tt(f[["ff.linear_in.weight"]]),
    ff_out = tt(f[["ff.linear_out.weight"]]),
    ff_context_in = tt(f[["ff_context.linear_in.weight"]]),
    ff_context_out = tt(f[["ff_context.linear_out.weight"]])
)

blk <- yq_flux2_double_block(heads = 24L, head_dim = 128L)
fj <- anvl::jit(blk)
out <- fj(f$h, f$c, f$mod_img, f$mod_txt, f$cos, f$sin, w)

c_got <- as.array(out[[1]]); c_want <- as.array(f$c_out)
h_got <- as.array(out[[2]]); h_want <- as.array(f$h_out)
mc <- max(abs(c_got - c_want)); mh <- max(abs(h_got - h_want))
cat(sprintf("flux2 double block parity: c max %.2e, h max %.2e\n", mc, mh))
expect_true(mh < 5e-3)
expect_true(mc < 5e-3)
expect_true(mean(abs(h_got - h_want)) < 5e-5)
expect_true(mean(abs(c_got - c_want)) < 5e-5)
