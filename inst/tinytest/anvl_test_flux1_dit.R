# Parity: yq_flux1_transformer (jitted, 2 double + 2 single blocks,
# random-init weights) vs the torch reference fixture from
# tools/gen_fixture_flux1_dit.R. Self-contained: the fixture carries the
# state dict, inputs, and output, so no external checkpoint is needed.
# Small config, CPU f32.

if (!tinytest::at_home()) {
    exit_file("at_home only")
}
if (!requireNamespace("anvl", quietly = TRUE) ||
    !requireNamespace("yunque", quietly = TRUE)) {
    exit_file("anvl not installed")
}
fixture <- file.path(Sys.getenv("HOME"),
                     ".local/share/R/yunque/fixtures/flux1_dit.safetensors")
if (!file.exists(fixture)) {
    exit_file("fixture missing; run tools/gen_fixture_flux1_dit.R")
}

num_layers <- 2L
num_single_layers <- 2L
heads <- 4L
head_dim <- 16L

f <- anvl::nv_read(fixture)
w <- yq_flux1_load_weights(fixture, num_layers = num_layers,
                           num_single_layers = num_single_layers,
                           device = "cpu")

fj <- anvl::jit(yq_flux1_transformer(
    num_layers = num_layers, num_single_layers = num_single_layers,
    heads = heads, head_dim = head_dim, mlp_ratio = 4.0,
    precision = "highest"))

out <- fj(f[["input.latents"]], f[["input.text_embeds"]],
          f[["input.pooled"]], f[["input.time_sin"]],
          f[["input.cos"]], f[["input.sin"]], w)

got <- as.array(out)
want <- as.array(f[["output"]])
max_abs <- max(abs(got - want))
scale <- max(abs(want))
cor_val <- cor(as.vector(got), as.vector(want))
cat(sprintf("flux1 dit parity: max %.3e mean %.3e cor %.6f (out scale %.3f)\n",
            max_abs, mean(abs(got - want)), cor_val, scale))

expect_equal(dim(got), c(1L, 16L, 8L))
expect_true(cor_val > 0.999999)
# f32 tolerance relative to output scale (accumulated over the block stack)
expect_true(max_abs < 1e-4 * scale)
expect_true(mean(abs(got - want)) < 1e-5 * scale)

# Host-side sinusoid matches the reference's internal timestep embedding
# (f32 tolerance: cos/sin computed in R f64 then cast vs torch f32).
ts_host <- as.array(yq_flux1_time_embed(
    as.numeric(as.array(f[["input.timestep"]])), dim = 256L))
expect_true(max(abs(ts_host - as.array(f[["input.time_sin"]]))) < 1e-4)
