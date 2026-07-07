# Tests for the LTX-2.3 single-file checkpoint reader.
# Uses a tiny fake checkpoint written with official-style key names and
# model_version metadata; no model downloads required.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}
if (!requireNamespace("safetensors", quietly = TRUE)) {
  exit_file("safetensors not installed")
}

library(diffuseR)

make_fake_checkpoint <- function(path, version = "2.3.0") {
  tensors <- list(
    # DiT group
    "model.diffusion_model.patchify_proj.weight" =
      torch::torch_randn(8, 4),
    "model.diffusion_model.patchify_proj.bias" =
      torch::torch_randn(8),
    "model.diffusion_model.scale_shift_table" =
      torch::torch_randn(6, 8),
    "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight" =
      torch::torch_randn(8, 8),
    # Connector group (inside and outside the diffusion_model prefix)
    "model.diffusion_model.video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight" =
      torch::torch_randn(8, 8),
    "model.diffusion_model.audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight" =
      torch::torch_randn(8, 8),
    "text_embedding_projection.video_aggregate_embed.weight" =
      torch::torch_randn(8, 12),
    "text_embedding_projection.audio_aggregate_embed.weight" =
      torch::torch_randn(4, 12),
    # VAE / audio VAE / vocoder groups
    "vae.decoder.conv_in.conv.weight" = torch::torch_randn(4, 4, 3, 3, 3),
    "audio_vae.decoder.conv_in.weight" = torch::torch_randn(4, 2, 3, 3),
    "vocoder.vocoder.conv_in.weight" = torch::torch_randn(4, 4, 7),
    "vocoder.bwe_generator.conv_in.weight" = torch::torch_randn(4, 4, 7),
    "vocoder.mel_stft.mel_basis" = torch::torch_randn(64, 257)
  )
  metadata <- list(model_version = version)
  safetensors::safe_save_file(tensors, path, metadata = metadata)
  invisible(tensors)
}

tmp <- tempfile(fileext = ".safetensors")
on.exit(unlink(tmp), add = TRUE)
tensors <- make_fake_checkpoint(tmp)

# --- Open + version gate -----------------------------------------------------

ckpt <- ltx23_open_checkpoint(tmp)
expect_inherits(ckpt, "ltx23_checkpoint")
expect_equal(ckpt$version, "2.3.0")
expect_equal(sort(ckpt$keys), sort(names(tensors)))

# Wrong version is rejected
tmp_old <- tempfile(fileext = ".safetensors")
on.exit(unlink(tmp_old), add = TRUE)
make_fake_checkpoint(tmp_old, version = "2.0.1")
expect_error(ltx23_open_checkpoint(tmp_old), pattern = "2\\.3")

# Version check can be disabled
ckpt_old <- ltx23_open_checkpoint(tmp_old, require_version = NULL)
expect_equal(ckpt_old$version, "2.0.1")

# Missing file
expect_error(ltx23_open_checkpoint(tempfile()), pattern = "not found")

# --- Key group split ---------------------------------------------------------

groups <- ltx23_split_keys(ckpt$keys)
expect_equal(length(groups$dit), 4L)
expect_equal(length(groups$connectors), 4L)
expect_equal(length(groups$vae), 1L)
expect_equal(length(groups$audio_vae), 1L)
expect_equal(length(groups$vocoder), 3L)
expect_equal(length(groups$other), 0L)

# Every key lands in exactly one group
expect_equal(sort(unlist(groups, use.names = FALSE)), sort(ckpt$keys))

# Census matches
cen <- ltx23_census(ckpt)
expect_equal(cen$keys[cen$group == "vocoder"], 3L)

# --- Streaming group load ----------------------------------------------------

toy <- torch::nn_module(
  "toy",
  initialize = function() {
    self$patchify_proj <- torch::nn_linear(4, 8)
    self$scale_shift_table <- torch::nn_parameter(torch::torch_zeros(6, 8))
    self$blocks <- torch::nn_module_list(list(
      torch::nn_module(
        "toy_block",
        initialize = function() {
          self$to_q <- torch::nn_linear(8, 8, bias = FALSE)
        }
      )()
    ))
  }
)()

map_dit_toy <- function(key) {
  key <- sub("^model\\.diffusion_model\\.", "", key)
  key <- sub("^transformer_blocks\\.0\\.attn1\\.", "blocks.0.", key)
  key
}

res <- ltx23_load_group(ckpt, groups$dit, toy, map_key = map_dit_toy, verbose = FALSE)
expect_equal(length(res$unmapped), 0L)
expect_equal(length(res$unfilled), 0L)
expect_equal(length(res$skipped), 0L)

# Values actually copied
expect_equal(
  as.numeric(torch::torch_sum(toy$patchify_proj$weight)),
  as.numeric(torch::torch_sum(tensors[["model.diffusion_model.patchify_proj.weight"]])),
  tolerance = 1e-5
)

# copy_ converts dtype: load into a float64 module
toy64 <- torch::nn_module(
  "toy64",
  initialize = function() {
    self$w <- torch::nn_parameter(torch::torch_zeros(8, 4, dtype = torch::torch_float64()))
  }
)()
res64 <- ltx23_load_group(
  ckpt, "model.diffusion_model.patchify_proj.weight", toy64,
  map_key = function(key) "w", verbose = FALSE
)
expect_equal(length(res64$unmapped), 0L)
expect_equal(toy64$w$dtype$.type(), "Double")

# Unmapped keys are reported, not fatal
res_bad <- ltx23_load_group(
  ckpt, groups$vae, toy,
  map_key = identity, verbose = FALSE
)
expect_equal(length(res_bad$unmapped), 1L)

# Mapper can deliberately skip keys with NA
res_skip <- ltx23_load_group(
  ckpt, groups$vae, toy,
  map_key = function(key) NA_character_, verbose = FALSE
)
expect_equal(length(res_skip$skipped), 1L)
expect_equal(length(res_skip$unmapped), 0L)

# Shape mismatch is a hard error
expect_error(
  ltx23_load_group(
    ckpt, "model.diffusion_model.scale_shift_table", toy,
    map_key = function(key) "patchify_proj.weight", verbose = FALSE
  ),
  pattern = "Shape mismatch"
)

# --- Real checkpoint census (local only, needs the 46GB file) ----------------

if (at_home()) {
  real <- tryCatch(
    hfhub::hub_download(
      "Lightricks/LTX-2.3", "ltx-2.3-22b-distilled-1.1.safetensors",
      local_files_only = TRUE
    ),
    error = function(e) NULL
  )
  if (!is.null(real)) {
    rc <- ltx23_open_checkpoint(real)
    expect_equal(rc$version, "2.3.0")
    expect_equal(length(rc$keys), 5947L)
    rg <- ltx23_split_keys(rc$keys)
    expect_equal(length(rg$other), 0L)
    expect_equal(length(rg$vocoder), 1227L)
    expect_true(!is.null(rc$config$transformer$num_layers))
  }
}
