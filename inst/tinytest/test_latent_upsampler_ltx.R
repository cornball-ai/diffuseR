# LTX latent upsamplers vs diffusers LTX2LatentUpsamplerModel
# (fixtures: tools/gen_fixtures_latent_upsampler.py)

library(diffuseR)

fx_path <- system.file("tinytest", "fixtures", "ltx_latent_upsampler.safetensors",
                       package = "diffuseR")
if (!nzchar(fx_path)) {
    fx_path <- file.path("fixtures", "ltx_latent_upsampler.safetensors")
}
if (!file.exists(fx_path) || !requireNamespace("safetensors", quietly = TRUE)) {
    exit_file("latent upsampler fixtures or safetensors unavailable")
}
fx <- safetensors::safe_load_file(fx_path, framework = "torch")

load_tag <- function(tag) {
    prefix <- paste0(tag, ".")
    keys <- substring(grep(paste0("^", tag, "\\."), names(fx), value = TRUE),
                      nchar(prefix) + 1L)
    get <- function(k) fx[[paste0(prefix, k)]]
    model <- diffuseR:::.ltx_upsampler_for(get, keys)
    dests <- c(model$named_parameters(), model$named_buffers())
    expect_true(setequal(names(dests), keys), info = paste(tag, "key set"))
    torch::with_no_grad({
        for (k in keys) {
            dests[[k]]$copy_(get(k))
        }
    })
    model$eval()
    model
}

for (tag in c("spatial", "temporal")) {
    model <- load_tag(tag)
    expect_identical(model$temporal, tag == "temporal")
    x <- fx[[paste0(tag, "_x")]]
    ref <- fx[[paste0(tag, "_out")]]
    out <- torch::with_no_grad(model(x))
    expect_equal(as.integer(out$shape), as.integer(ref$shape),
                 info = paste(tag, "shape"))
    expect_true((out - ref)$abs()$max()$item() < 1e-5,
                info = paste(tag, "parity"))
}

# Temporal geometry: F latent frames -> 2F - 1 (8k + 1 pixel frames -> 16k + 1)
expect_equal(as.integer(fx$temporal_out$shape[3]),
             2L * as.integer(fx$temporal_x$shape[3]) - 1L)
