# Parity: yq_ltx_dit (jitted anvl LTX-2.3 audio-video DiT) vs the torch
# reference fixture from tools/gen_fixture_ltx_dit.R (random-init weights,
# small config). CPU f32. The DiT is driven entirely from host-side inputs
# (timestep sinusoids, split-RoPE cos/sin tables, additive text masks all
# computed in base R), so a green run also validates the host-side helpers
# end-to-end. Explicit cross-checks compare each host-side RoPE table to
# the reference tables saved in the fixture.

if (!tinytest::at_home()) {
    exit_file("at_home only")
}
if (!requireNamespace("anvl", quietly = TRUE) ||
    !requireNamespace("yunque", quietly = TRUE)) {
    exit_file("anvl not installed")
}
fixture <- file.path(Sys.getenv("HOME"),
                     ".local/share/R/diffuseR/fixtures/ltx_dit.safetensors")
if (!file.exists(fixture)) {
    exit_file("fixture missing (run tools/gen_fixture_ltx_dit.R 0)")
}

# fixture config (must match tools/gen_fixture_ltx_dit.R)
heads <- 3L; head_dim <- 8L          # video: inner 24
a_heads <- 2L; a_head_dim <- 6L      # audio: inner 12
ainner <- a_heads * a_head_dim
n_layers <- 2L
nf <- 2L; H <- 3L; W <- 2L; audio_nf <- 5L

f <- anvl::nv_read(fixture)
w <- yq_ltx_dit_load_weights(fixture, isolate = FALSE, device = "cpu")

# ---- host-side timestep sinusoids from the saved (raw) timesteps/sigmas ----
flat <- function(x) as.numeric(t(as.array(x)))   # row-major flatten
sins <- list(
    time         = yq_ltx_time_sinusoid(flat(f$timestep)),
    audio_time   = yq_ltx_time_sinusoid(flat(f$audio_timestep)),
    prompt       = yq_ltx_time_sinusoid(flat(f$sigma)),
    audio_prompt = yq_ltx_time_sinusoid(flat(f$audio_sigma))
)

# ---- host-side split-RoPE tables ----
vr  <- yq_ltx_video_rope(nf, H, W, num_heads = heads, head_dim = head_dim)
ar  <- yq_ltx_audio_rope(audio_nf, num_heads = a_heads, head_dim = a_head_dim)
vcr <- yq_ltx_video_cross_rope(nf, H, W, num_heads = heads, cross_dim = ainner)
ropes <- list(v_cos = vr$cos, v_sin = vr$sin, a_cos = ar$cos, a_sin = ar$sin,
              vca_cos = vcr$cos, vca_sin = vcr$sin,
              aca_cos = ar$cos, aca_sin = ar$sin)   # audio cross == audio self

# ---- host-side additive text masks ([B,S] -> [B,1,1,S]) ----
act <- list(hidden = f$hidden, audio_hidden = f$audio_hidden,
            enc = f$enc, audio_enc = f$audio_enc,
            enc_mask = yq_ltx_text_mask(as.array(f$enc_mask)),
            audio_enc_mask = yq_ltx_text_mask(as.array(f$aenc_mask)),
            self_mask = NULL)

dit <- anvl::jit(yq_ltx_dit(heads = heads, head_dim = head_dim,
                            a_heads = a_heads, a_head_dim = a_head_dim,
                            num_layers = n_layers, isolate = FALSE,
                            precision = "highest"))
out <- dit(act, sins, ropes, w)

parity <- function(got, want, name) {
    g <- as.array(got); wv <- as.array(want)
    max_abs <- max(abs(g - wv)); sd_out <- sd(as.vector(wv))
    correlation <- cor(as.vector(g), as.vector(wv))
    cat(sprintf("LTX DiT %-6s parity: max %.2e  mean %.2e  cor %.6f  (out sd %.3f)\n",
                name, max_abs, mean(abs(g - wv)), correlation, sd_out))
    expect_equal(dim(g), dim(wv))
    expect_true(correlation > 0.999999)
    expect_true(max_abs < 1e-4 * max(1, sd_out))
    expect_true(mean(abs(g - wv)) < 1e-5 * max(1, sd_out))
}
parity(out$video, f$video_out, "video")
parity(out$audio, f$audio_out, "audio")

# ---- host-side RoPE cross-check against the reference tables ----
rmax <- max(
    max(abs(as.array(vr$cos)  - as.array(f$v_cos))),
    max(abs(as.array(vr$sin)  - as.array(f$v_sin))),
    max(abs(as.array(ar$cos)  - as.array(f$a_cos))),
    max(abs(as.array(ar$sin)  - as.array(f$a_sin))),
    max(abs(as.array(vcr$cos) - as.array(f$vca_cos))),
    max(abs(as.array(vcr$sin) - as.array(f$vca_sin))),
    max(abs(as.array(ar$cos)  - as.array(f$aca_cos))),
    max(abs(as.array(ar$sin)  - as.array(f$aca_sin)))
)
cat(sprintf("LTX RoPE host-side parity: max %.2e\n", rmax))
expect_true(rmax < 1e-6)
