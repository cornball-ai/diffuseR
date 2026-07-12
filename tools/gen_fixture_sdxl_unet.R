# Generate the SDXL UNet parity fixture (torch reference) with
# RANDOM-INIT weights (architecture parity; no checkpoint download).
# Instantiates diffuseR::unet_sdxl_native with the production SDXL config
# under a fixed seed, saves its state_dict (weights) to one f32
# safetensors and the random inputs + output to another. The anvl test
# reloads the SAME state_dict via yq_sdxl_unet_load_weights and feeds
# these inputs.
#
# Usage: /home/troy/diffuseR-sdxl-lib/ranvl tools/gen_fixture_sdxl_unet.R

suppressMessages(library(torch))

fixture_dir <- file.path(Sys.getenv("HOME"), ".local/share/R/diffuseR/fixtures")
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)
weights_file <- file.path(fixture_dir, "sdxl_unet_weights.safetensors")
io_file <- file.path(fixture_dir, "sdxl_unet_io.safetensors")

# Production SDXL config (channels 320/640/1280, transformer depth
# 0/2/10, cross-attn 2048, add-embedding: pooled 1280 + 6*256 time-ids).
torch_manual_seed(0)
set.seed(0)
unet <- diffuseR::unet_sdxl_native(
    in_channels = 4L, out_channels = 4L,
    block_out_channels = c(320L, 640L, 1280L),
    layers_per_block = 2L,
    transformer_layers_per_block = c(0L, 2L, 10L),
    cross_attention_dim = 2048L, attention_head_dim = 64L,
    addition_embed_dim = 1280L, addition_time_embed_dim = 256L)
unet$eval()

# Save the random-init state_dict (f32) as the weights fixture the anvl
# loader reads back. The full SDXL UNet is ~10 GB f32, and the R
# `safetensors` writer overflows int32 when computing data offsets past
# 2 GB (corrupting the header). yunque's reader handles int64 offsets
# fine (jsonlite parses them as doubles, seek() accepts doubles), so we
# write the file directly with an int64-safe, row-major F32 writer.
save_st_f32 <- function(tensors, path) {
    keys <- names(tensors)
    shapes <- lapply(tensors, function(t) as.integer(t$shape))
    nbytes <- vapply(shapes, function(s) prod(as.double(s)) * 4, numeric(1))
    ends <- cumsum(nbytes)
    starts <- c(0, ends[-length(ends)])
    header <- list()
    for (i in seq_along(keys)) {
        header[[keys[i]]] <- list(
            dtype = "F32",
            shape = as.list(shapes[[i]]),
            data_offsets = list(starts[i], ends[i]))
    }
    jraw <- charToRaw(jsonlite::toJSON(header, auto_unbox = TRUE, digits = NA))
    con <- file(path, "wb")
    on.exit(close(con))
    # 8-byte little-endian header length (value < 2^31, high word zero).
    writeBin(length(jraw), con, size = 4L, endian = "little")
    writeBin(0L, con, size = 4L, endian = "little")
    writeBin(jraw, con)
    for (k in keys) {
        v <- as.array(tensors[[k]]$to(dtype = torch_float32())$contiguous()$reshape(c(-1L)))
        writeBin(as.numeric(v), con, size = 4L, endian = "little")
    }
}

sd <- unet$state_dict()
save_st_f32(sd, weights_file)
cat(sprintf("weights: %s (%.2f GB, %d tensors)\n", weights_file,
            file.size(weights_file) / 1e9, length(sd)))

# Random inputs (small spatial: H = W = 16).
set.seed(7)
torch_manual_seed(7)
H <- 16L; W <- 16L; seq_len <- 16L; cross_dim <- 2048L
sample <- torch_randn(1L, 4L, H, W)
context <- torch_randn(1L, seq_len, cross_dim)      # encoder_hidden_states
text_embeds <- torch_randn(1L, 1280L)               # pooled text embeds
time_ids <- torch_tensor(matrix(c(1024, 1024, 256, 128, 768, 512), nrow = 1L),
                         dtype = torch_float32())    # [1, 6]
timestep_val <- 500
timestep <- torch_tensor(timestep_val, dtype = torch_float32())$reshape(1L)

# The parameter-free sinusoids the UNet computes internally, saved so the
# anvl port can feed them directly (flip_sin_to_cos = TRUE,
# downscale_freq_shift = 0).
t_sin <- diffuseR:::timestep_embedding(timestep, 320L, flip_sin_to_cos = TRUE,
                                       downscale_freq_shift = 0L)
time_ids_sin <- diffuseR:::timestep_embedding(time_ids$flatten(), 256L,
                                              flip_sin_to_cos = TRUE,
                                              downscale_freq_shift = 0L)
time_ids_sin <- time_ids_sin$reshape(c(1L, -1L))    # [1, 1536]

out <- with_no_grad(unet(sample, timestep, context, text_embeds, time_ids))

safetensors::safe_save_file(list(
    sample = sample$contiguous(),
    t_sin = t_sin$contiguous(),
    time_ids_sin = time_ids_sin$contiguous(),
    text_embeds = text_embeds$contiguous(),
    context = context$contiguous(),
    timestep = timestep$contiguous(),
    time_ids = time_ids$contiguous(),
    out = out$contiguous()
), io_file)

cat(sprintf("io: %s (%.2f MB)\n", io_file, file.size(io_file) / 1e6))
cat(sprintf("out shape %s sd %.4f range [%.3f, %.3f]\n",
            paste(dim(out), collapse = "x"), out$std()$item(),
            out$min()$item(), out$max()$item()))
