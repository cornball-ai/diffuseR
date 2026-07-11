# End-to-end latents -> pixels parity: unpack + BN denorm + unpatchify
# (yq_flux2_vae_prepare) then the VAE decoder, vs the torch reference in
# flux2_pixels.safetensors, starting from the sample loop's final
# latents. Only the VAE weights load here (the encoder/DiT ran when the
# sample fixture was made), so this is fast.

if (!tinytest::at_home()) {
    exit_file("at_home only")
}
if (!requireNamespace("anvl", quietly = TRUE) ||
    !requireNamespace("yunque", quietly = TRUE)) {
    exit_file("anvl not installed")
}
vae <- Sys.glob(file.path(
    Sys.getenv("HOME"),
    ".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B",
    "snapshots/*/vae/diffusion_pytorch_model.safetensors"))[1]
sample <- file.path(Sys.getenv("HOME"),
                    ".local/share/R/yunque/fixtures/flux2_sample.safetensors")
pixels <- file.path(Sys.getenv("HOME"),
                    ".local/share/R/yunque/fixtures/flux2_pixels.safetensors")
if (is.na(vae) || !file.exists(vae) || !file.exists(sample) ||
    !file.exists(pixels)) {
    exit_file("checkpoint or fixtures missing")
}

w <- yq_flux2_load_vae(vae, device = "cpu")
final <- as.array(anvl::nv_read(sample)$final)     # [1, 256, 128]
want <- as.array(anvl::nv_read(pixels)$pix)        # [1, 3, 256, 256]

z <- yq_flux2_vae_prepare(final, 16L, 16L,
                          as.numeric(w$bn_mean), as.numeric(w$bn_var),
                          device = "cpu")
expect_equal(anvl::shape(z), c(1L, 32L, 32L, 32L))

pix <- anvl::jit(function(zz) yq_flux2_vae_decode(zz, w))(z)
got <- as.array(pix)
max_abs <- max(abs(got - want))
cat(sprintf("latents->pixels parity: max %.2e mean %.2e cor %.6f\n",
            max_abs, mean(abs(got - want)),
            cor(as.vector(got), as.vector(want))))
expect_equal(dim(got), c(1L, 3L, 256L, 256L))
expect_true(max_abs < 1e-3)
expect_true(mean(abs(got - want)) < 1e-4)
