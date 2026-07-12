# Parity: yq_zimage_generate (full anvl Z-Image-Turbo text-to-image, REAL
# weights, fp8 transformer dequantized to f32 + bf16 Qwen3 upcast) vs the
# torch reference from tools/gen_fixture_zimage_e2e.R. Feeds the same
# initial noise + token ids; compares the Qwen3 caption features, the
# final latents, and the decoded pixels at f32 tolerance (correlation ~1,
# scale-relative max-abs). Heavy (loads ~24 GB of f32 weights across
# phases), so at_home + gated on the real artifacts being present.

if (!tinytest::at_home()) {
    exit_file("at_home only")
}
if (!requireNamespace("anvl", quietly = TRUE) ||
    !requireNamespace("yunque", quietly = TRUE)) {
    exit_file("anvl not installed")
}

## ---- fixture config (MUST match tools/gen_fixture_zimage_e2e.R) ----
height <- 128L
width <- 128L
steps <- 4L
shift <- 3.0
penult_layer <- 35L

fixture_dir <- file.path(Sys.getenv("HOME"), ".local/share/R/diffuseR/fixtures")
fixture_path <- file.path(fixture_dir, "zimage_e2e.safetensors")
vae_path <- file.path(fixture_dir, "zimage_e2e_vae.safetensors")
dit_dir <- file.path(tools::R_user_dir("diffuseR", "data"), "zimage-turbo-fp8")
qwen_dir <- Sys.glob(file.path(
    Sys.getenv("HOME"),
    ".cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo",
    "snapshots/*/text_encoder"))
qwen_dir <- if (length(qwen_dir)) qwen_dir[[1]] else ""

if (!file.exists(fixture_path) || !file.exists(vae_path) ||
    !file.exists(file.path(dit_dir, "manifest.json")) || !dir.exists(qwen_dir)) {
    exit_file("fixture or real weights missing (run gen_fixture_zimage_e2e.R)")
}

## ---- load the reference fixture ----
f <- anvl::nv_read(fixture_path)
input_ids <- matrix(as.integer(round(as.array(f$input_ids))), nrow = 1L)
attention_mask <- matrix(as.integer(round(as.array(f$attention_mask))), nrow = 1L)

## ---- run the anvl pipeline on the same noise + ids ----
res <- yq_zimage_generate(
    input_ids = input_ids, attention_mask = attention_mask, noise = f$noise,
    height = height, width = width, dit_dir = dit_dir, qwen_dir = qwen_dir,
    vae_path = vae_path, steps = steps, shift = shift,
    penult_layer = penult_layer, decode = TRUE, device = "cpu",
    precision = "highest", verbose = TRUE)

report <- function(label, got, want) {
    got <- as.vector(got)
    want <- as.vector(want)
    max_abs <- max(abs(got - want))
    scale <- max(sd(want), max(abs(want)), 1e-8)
    correlation <- cor(got, want)
    cat(sprintf("%-14s max %.3e  rel %.3e  mean %.3e  cor %.6f  (scale %.3f)\n",
                label, max_abs, max_abs / scale, mean(abs(got - want)),
                correlation, scale))
    list(max_abs = max_abs, rel = max_abs / scale, cor = correlation)
}

## ---- Qwen3 caption features (debug the text stage first) ----
cap_got <- as.array(res$cap_feats)[1, , ]        # [n_real, 2560]
cap_want <- as.array(f$cap_feats)                # [n_real, 2560]
expect_equal(dim(cap_got), dim(cap_want))
c_cap <- report("cap_feats", cap_got, cap_want)
expect_true(c_cap$cor > 0.9999)
expect_true(c_cap$rel < 5e-3)                    # 35-layer f32 rounding at sd ~35

## ---- final latents ----
lat_got <- as.array(res$latents)
lat_want <- as.array(f$latents)
expect_equal(dim(lat_got), dim(lat_want))
c_lat <- report("latents", lat_got, lat_want)
expect_true(c_lat$cor > 0.9999)
expect_true(c_lat$rel < 5e-3)

## ---- decoded pixels ----
pix_got <- as.array(res$pixels)
pix_want <- as.array(f$pixels)
expect_equal(dim(pix_got), dim(pix_want))
c_pix <- report("pixels", pix_got, pix_want)
expect_true(c_pix$cor > 0.9999)
expect_true(c_pix$rel < 5e-3)
