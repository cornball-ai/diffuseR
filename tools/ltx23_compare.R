# treesitR coverage report: diffusers LTX-2 reference vs the diffuseR port.
#
# Parses the Python reference files and the R port with tree-sitter,
# extracts class/function inventories, and prints a coverage table using
# a curated mapping (which also documents deliberate omissions).
#
# Usage: r tools/ltx23_compare.R

library(treesitR)

`%||%` <- function(x, y) if (is.null(x)) y else x

py_entities <- function(path) {
  src <- paste(readLines(path, warn = FALSE), collapse = "\n")
  parser <- ts_parser_new()
  ts_parser_set_language(parser, ts_language_python())
  tree <- ts_parse(parser, src)
  root <- ts_tree_root_node(tree)

  out <- character(0)
  # Top-level classes and functions only (methods are per-class detail)
  for (i in seq_len(ts_node_named_child_count(root))) {
    node <- ts_node_named_child(root, i - 1L)
    if (ts_node_type(node) == "decorated_definition") {
      node <- ts_node_child_by_field(node, "definition")
    }
    type <- ts_node_type(node)
    if (type %in% c("class_definition", "function_definition")) {
      name <- ts_node_text(ts_node_child_by_field(node, "name"))
      prefix <- if (type == "class_definition") "class" else "def"
      out[[length(out) + 1L]] <- paste(prefix, name)
    }
  }
  unique(out)
}

r_entities <- function(path) {
  src <- paste(readLines(path, warn = FALSE), collapse = "\n")
  parser <- ts_parser_new()
  ts_parser_set_language(parser, ts_language_r())
  tree <- ts_parse(parser, src)
  root <- ts_tree_root_node(tree)

  out <- character(0)
  for (i in seq_len(ts_node_named_child_count(root))) {
    node <- ts_node_named_child(root, i - 1L)
    if (ts_node_type(node) == "binary_operator") {
      lhs <- ts_node_child_by_field(node, "lhs")
      if (!is.null(lhs) && ts_node_type(lhs) == "identifier") {
        out[[length(out) + 1L]] <- ts_node_text(lhs)
      }
    }
  }
  unique(out)
}

ref_dir <- "ref/upstream/diffusers/src/diffusers"
py_files <- c(
  transformer = file.path(ref_dir, "models/transformers/transformer_ltx2.py"),
  video_vae = file.path(ref_dir, "models/autoencoders/autoencoder_kl_ltx2.py"),
  audio_vae = file.path(ref_dir, "models/autoencoders/autoencoder_kl_ltx2_audio.py"),
  connectors = file.path(ref_dir, "pipelines/ltx2/connectors.py"),
  vocoder = file.path(ref_dir, "pipelines/ltx2/vocoder.py"),
  upsampler = file.path(ref_dir, "pipelines/ltx2/latent_upsampler.py")
)

# Curated mapping: reference entity -> R counterpart or a documented skip
MAPPING <- c(
  # transformer_ltx2.py
  "def apply_interleaved_rotary_emb" = "ltx23_apply_interleaved_rotary_emb",
  "def apply_split_rotary_emb" = "ltx23_apply_split_rotary_emb",
  "class LTX2AdaLayerNormSingle" = "ltx23_ada_layer_norm_single",
  "class LTX2AudioVideoAttnProcessor" = "ltx23_attention (inlined)",
  "class LTX2PerturbedAttnProcessor" = "ltx23_attention (inlined)",
  "class LTX2Attention" = "ltx23_attention",
  "class LTX2VideoTransformerBlock" = "ltx23_transformer_block",
  "class LTX2AudioVideoRotaryPosEmbed" = "ltx23_rotary_pos_embed",
  "class LTX2VideoTransformer3DModel" = "ltx23_transformer",
  "class AudioVisualModelOutput" = "SKIP: plain list return",
  # autoencoder_kl_ltx2.py
  "class PerChannelRMSNorm" = "ltx23_per_channel_rms_norm",
  "class LTX2VideoCausalConv3d" = "ltx23_causal_conv3d",
  "class LTX2VideoResnetBlock3d" = "ltx23_video_resnet_block3d",
  "class LTX2VideoDownsampler3d" = "ltx23_video_downsampler3d",
  "class LTX2VideoUpsampler3d" = "ltx23_video_upsampler3d",
  "class LTX2VideoDownBlock3D" = "ltx23_video_down_block3d",
  "class LTX2VideoMidBlock3d" = "ltx23_video_mid_block3d",
  "class LTX2VideoUpBlock3d" = "ltx23_video_up_block3d",
  "class LTX2VideoEncoder3d" = "ltx23_video_encoder3d",
  "class LTX2VideoDecoder3d" = "ltx23_video_decoder3d",
  "class AutoencoderKLLTX2Video" = "ltx23_video_vae",
  # autoencoder_kl_ltx2_audio.py
  "class LTX2AudioCausalConv2d" = "ltx23_audio_causal_conv2d",
  "class LTX2AudioPixelNorm" = "ltx23_per_channel_rms_norm (shared)",
  "class LTX2AudioAttnBlock" = "SKIP: mid_block_add_attention FALSE in 2.3",
  "class LTX2AudioResnetBlock" = "ltx23_audio_resnet_block",
  "class LTX2AudioDownsample" = "SKIP: encoder-only (t2v never encodes audio)",
  "class LTX2AudioUpsample" = "ltx23_audio_upsample",
  "class LTX2AudioAudioPatchifier" = "SKIP: pipeline packs latents directly",
  "class LTX2AudioEncoder" = "SKIP: encoder-only (t2v never encodes audio)",
  "class LTX2AudioDecoder" = "ltx23_audio_decoder",
  "class AutoencoderKLLTX2Audio" = "ltx23_audio_vae",
  # connectors.py
  "def per_layer_masked_mean_norm" = "SKIP: LTX-2.0 path (per_modality_projections)",
  "def per_token_rms_norm" = "ltx23_per_token_rms_norm",
  "class LTX2RotaryPosEmbed1d" = "ltx23_rotary_pos_embed_1d",
  "class LTX2TransformerBlock1d" = "ltx23_transformer_block_1d",
  "class LTX2ConnectorTransformer1d" = "ltx23_connector_transformer_1d",
  "class LTX2TextConnectors" = "ltx23_text_connectors",
  # vocoder.py
  "def kaiser_sinc_filter1d" = "ltx23_kaiser_sinc_filter1d",
  "class DownSample1d" = "ltx23_downsample1d",
  "class UpSample1d" = "ltx23_upsample1d",
  "class AntiAliasAct1d" = "ltx23_antialias_act1d",
  "class SnakeBeta" = "ltx23_snake_beta",
  "class ResBlock" = "ltx23_vocoder_resblock / ltx23_upsampler_res_block",
  "class LTX2Vocoder" = "ltx23_vocoder",
  "class CausalSTFT" = "ltx23_causal_stft",
  "class MelSTFT" = "ltx23_mel_stft",
  "class LTX2VocoderWithBWE" = "ltx23_vocoder_with_bwe",
  # latent_upsampler.py
  "class PixelShuffleND" = "nnf_pixel_shuffle (2D case only in 2.3)",
  "class BlurDownsample" = "SKIP: rational resampler off in 2.3",
  "class SpatialRationalResampler" = "SKIP: rational resampler off in 2.3",
  "class LTX2LatentUpsamplerModel" = "ltx23_latent_upsampler"
)

r_files <- list.files("R", pattern = "ltx23|txt2vid_ltx23", full.names = TRUE)
r_defined <- unlist(lapply(r_files, r_entities))

missing <- character(0)
cat(sprintf("%-45s %-50s %s\n", "REFERENCE", "R PORT", "STATUS"))
cat(strrep("-", 110), "\n")
for (section in names(py_files)) {
  cat("##", section, "\n")
  for (entity in py_entities(py_files[[section]])) {
    mapped <- if (entity %in% names(MAPPING)) MAPPING[[entity]] else NULL
    status <- if (is.null(mapped)) {
      missing <- c(missing, paste0(section, ": ", entity))
      "MISSING FROM MAPPING"
    } else if (startsWith(mapped, "SKIP")) {
      "skipped (documented)"
    } else {
      r_name <- strsplit(mapped, " ")[[1]][1]
      if (r_name %in% r_defined || grepl("nnf_", r_name)) "ported" else "MAPPED BUT NOT DEFINED"
    }
    cat(sprintf("%-45s %-50s %s\n", entity, mapped %||% "-", status))
  }
}

if (length(missing)) {
  cat("\nUNMAPPED reference entities:\n")
  for (m in missing) cat("  -", m, "\n")
  quit(status = 1)
}
cat("\nAll reference entities accounted for.\n")
