# Parity: full yq_flux2_transformer (jitted, all 25 blocks, real Klein
# weights) vs the torch reference fixture from
# tools/gen_fixture_flux2_forward.R. Loads the full checkpoint (~15.5 GB
# f32 in host RAM, ~110s) so it is at_home + checkpoint-gated.

if (!tinytest::at_home()) {
    exit_file("at_home only")
}
if (!requireNamespace("anvl", quietly = TRUE) ||
    !requireNamespace("yunque", quietly = TRUE)) {
    exit_file("anvl not installed")
}
ckpt <- file.path(
    Sys.getenv("HOME"),
    ".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B",
    "snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294",
    "transformer/diffusion_pytorch_model.safetensors")
fixture <- file.path(Sys.getenv("HOME"),
                     ".local/share/R/yunque/fixtures/flux2_forward.safetensors")
if (!file.exists(ckpt) || !file.exists(fixture)) {
    exit_file("checkpoint or fixture missing")
}

f <- anvl::nv_read(fixture)
w <- yq_flux2_load_weights(ckpt, device = "cpu")
tp <- yq_flux2_time_proj(0.7, device = "cpu")
rope <- yq_flux2_rope(64L, 16L, 16L, device = "cpu")

fj <- anvl::jit(yq_flux2_transformer(precision = "highest"))
out <- fj(f$latents, f$text_embeds, tp, rope$cos, rope$sin, w)

got <- as.array(out); want <- as.array(f$out)
max_abs <- max(abs(got - want))
cat(sprintf("full flux2 forward parity: max %.2e mean %.2e cor %.6f\n",
            max_abs, mean(abs(got - want)),
            cor(as.vector(got), as.vector(want))))
expect_equal(dim(got), c(1L, 256L, 128L))
expect_true(max_abs < 1e-3)
expect_true(mean(abs(got - want)) < 1e-4)
