# End-to-end parity: the full anvl FLUX.2 Klein text-to-latent pipeline
# (Qwen3 encoder -> DiT -> 4-step FlowMatch Euler loop) vs the torch
# reference from tools/gen_fixture_flux2_sample.R. Loads BOTH the text
# encoder (27 layers) and the DiT (~27 GB f32 host RAM, ~3 min), so
# at_home + checkpoint-gated.

if (!tinytest::at_home()) {
    exit_file("at_home only")
}
if (!requireNamespace("anvl", quietly = TRUE) ||
    !requireNamespace("yunque", quietly = TRUE)) {
    exit_file("anvl not installed")
}
snap <- file.path(
    Sys.getenv("HOME"),
    ".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B",
    "snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294")
te <- file.path(snap, "text_encoder")
dit <- file.path(snap, "transformer/diffusion_pytorch_model.safetensors")
fixture <- file.path(Sys.getenv("HOME"),
                     ".local/share/R/yunque/fixtures/flux2_sample.safetensors")
if (!dir.exists(te) || !file.exists(dit) || !file.exists(fixture)) {
    exit_file("checkpoint or fixture missing")
}

f <- anvl::nv_read(fixture)
ids0 <- matrix(as.integer(round(as.array(f$ids0))), nrow = 1L)
attn <- matrix(as.integer(round(as.array(f$attn))), nrow = 1L)
sigmas <- as.numeric(as.array(f$sigmas))
S_txt <- ncol(ids0)

we <- yq_qwen3_load_weights(te, n_layers = 27L, device = "cpu")
wd <- yq_flux2_load_weights(dit, device = "cpu")

# conditioning: tokens -> embeds -> mid-stack states
embeds <- yq_qwen3_embed(we$embed, ids0, device = "cpu")
rq <- yq_qwen3_rope(S_txt, 128L, 1e6, device = "cpu")
mask <- yq_qwen3_mask(attn, S_txt, 1L, device = "cpu")
text_embeds <- anvl::jit(yq_qwen3_encoder(precision = "highest"))(
    embeds, rq$cos, rq$sin, mask, we)

# denoise: 4-step FlowMatch Euler over the DiT
rope <- yq_flux2_rope(S_txt, 16L, 16L, device = "cpu")
final <- yq_flux2_sample(f$latents0, text_embeds, sigmas,
                         rope$cos, rope$sin, wd, device = "cpu")

got <- as.array(final); want <- as.array(f$final)
max_abs <- max(abs(got - want))
cat(sprintf("e2e text->latent parity: max %.2e mean %.2e cor %.6f\n",
            max_abs, mean(abs(got - want)),
            cor(as.vector(got), as.vector(want))))
expect_equal(dim(got), c(1L, 256L, 128L))
expect_true(max_abs < 1e-3)
expect_true(mean(abs(got - want)) < 1e-4)

# sigma schedule matches the pipeline's (host-side, deterministic)
sig_yq <- yq_flux2_sigmas(256L, 4L)
expect_true(max(abs(sig_yq - sigmas)) < 1e-6)
