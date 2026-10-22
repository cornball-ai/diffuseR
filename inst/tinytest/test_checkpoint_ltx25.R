# ltx25_open_checkpoint(): the split LTX-2.5 files presented as one
# ltx23_checkpoint under the 2.3 key names.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}
if (!requireNamespace("safetensors", quietly = TRUE)) {
  exit_file("safetensors not installed")
}

library(diffuseR)

local({
  dir <- tempfile("ltx25-ckpt-")
  dir.create(dir)
  on.exit(unlink(dir, recursive = TRUE), add = TRUE)
  path <- function(n) file.path(dir, n)
  save <- function(tensors, name, version = "2.5.0", config = NULL) {
    meta <- list(model_version = version)
    if (!is.null(config)) {
      meta$config <- as.character(jsonlite::toJSON(config, auto_unbox = TRUE))
    }
    if (is.na(version)) meta$model_version <- NULL
    safetensors::safe_save_file(tensors, path(name), metadata = meta)
  }
  t <- function(v) torch::torch_tensor(v)

  save(list("model.diffusion_model.patchify_proj.weight" = t(c(1, 2)),
            "model.diffusion_model.video_embeddings_connector.learnable_registers" = t(3)),
       "dit.safetensors",
       config = list(transformer = list(ff_bias = FALSE), scheduler = list(a = 1)))
  save(list("model.layers.0.mlp.up_proj.weight" = t(9),
            "text_embedding_projection.video_aggregate_embed.weight" = t(c(4, 5))),
       "te.safetensors", version = NA)
  save(list("decoder.up_blocks.0.conv.weight" = t(6),
            "per_channel_statistics.mean-of-means" = t(7)),
       "vae.safetensors", config = list(vae = list(latent_channels = 128L)))
  save(list("audio_vae.decoder.conv_in.weight" = t(8),
            "vocoder.vocoder.conv_in.weight" = t(10)),
       "audio.safetensors",
       config = list(audio_vae = list(x = 1), vocoder = list(y = 2)))

  ck <- ltx25_open_checkpoint(path("dit.safetensors"), path("te.safetensors"),
                              path("vae.safetensors"), path("audio.safetensors"))
  expect_inherits(ck, "ltx23_checkpoint")
  expect_equal(ck$version, "2.5.0")
  expect_null(ck$format)

  groups <- ltx23_split_keys(ck$keys)
  expect_equal(groups$dit, "model.diffusion_model.patchify_proj.weight")
  expect_equal(sort(groups$connectors),
               sort(c("model.diffusion_model.video_embeddings_connector.learnable_registers",
                      "text_embedding_projection.video_aggregate_embed.weight")))
  expect_equal(sort(groups$vae), sort(c("vae.decoder.up_blocks.0.conv.weight",
                                        "vae.per_channel_statistics.mean-of-means")))
  expect_equal(groups$audio_vae, "audio_vae.decoder.conv_in.weight")
  expect_equal(groups$vocoder, "vocoder.vocoder.conv_in.weight")
  expect_equal(length(groups$other), 0L)
  # The encoder's own weights are not part of the pipeline checkpoint
  expect_false(any(grepl("^model\\.layers", ck$keys)))

  # Each key reads from its own file
  get <- function(k) as.numeric(ck$handle$get_tensor(k))
  expect_equal(get("text_embedding_projection.video_aggregate_embed.weight"), c(4, 5))
  expect_equal(get("vae.per_channel_statistics.mean-of-means"), 7)
  expect_equal(get("vocoder.vocoder.conv_in.weight"), 10)
  expect_equal(get("model.diffusion_model.patchify_proj.weight"), c(1, 2))
  expect_error(ck$handle$get_tensor("nope"), "not found")
  # The VAE map strips the prefix the composite added
  expect_equal(ltx23_map_vae_key("vae.per_channel_statistics.mean-of-means"),
               "latents_mean")

  # Configs merge from their own files
  expect_false(ck$config$transformer$ff_bias)
  expect_equal(ck$config$vae$latent_channels, 128L)
  expect_equal(ck$config$vocoder$y, 2L)

  # Audio is optional
  ck2 <- ltx25_open_checkpoint(path("dit.safetensors"), path("te.safetensors"),
                               path("vae.safetensors"))
  expect_equal(length(ltx23_split_keys(ck2$keys)$vocoder), 0L)

  # Wrong files are refused
  save(list("model.diffusion_model.patchify_proj.weight" = t(1)), "dit23.safetensors",
       version = "2.3.0")
  expect_error(ltx25_open_checkpoint(path("dit23.safetensors"), path("te.safetensors"),
                                     path("vae.safetensors")), "2\\.5")
  save(list("decoder.transformer_blocks.0.w" = t(1)), "vaediff.safetensors")
  expect_error(ltx25_open_checkpoint(path("dit.safetensors"), path("te.safetensors"),
                                     path("vaediff.safetensors")), "convolutional")
  expect_error(ltx25_open_checkpoint(path("dit.safetensors"), path("vae.safetensors"),
                                     path("vae.safetensors")), "text_embedding_projection")
})

# An encoder already on the CPU encodes there; staging or the GPU decides otherwise
m <- torch::nn_linear(2, 2)
expect_equal(diffuseR:::.ltx23_text_encode_device(m, "cuda"), "cpu")
expect_equal(diffuseR:::.ltx23_text_encode_device("path", "cuda"), "cuda")
