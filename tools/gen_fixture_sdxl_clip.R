# Generate the SDXL dual CLIP text-encoder parity fixture (torch
# reference) with RANDOM-INIT weights (architecture parity; no checkpoint
# download). Instantiates diffuseR::text_encoder_native (CLIP-L, 768/12/12,
# quick-GELU) and diffuseR::text_encoder2_native (bigG, 1280/32/20, exact
# GELU) under fixed seeds, randomizes the zero/unit-init params (position
# embeddings + LayerNorm affine) so every path is exercised, and reproduces
# the exact SDXL conditioning:
#   * context = concat(CLIP-L hidden_states[-2], bigG hidden_states[-2])
#               along the feature dim -> [B, seq, 2048]  (penultimate,
#               pre-final-LN, no clip_skip)
#   * pooled  = text_projection( final_LN(bigG)[EOS] )    -> [B, 1280]
# Each state_dict is written with the int64-safe row-major F32 writer
# (bigG's state_dict is ~2.8 GB f32, past the point where the R
# safetensors writer overflows int32 data offsets and corrupts the
# header). The anvl test reloads the SAME state_dicts via
# yq_sdxl_clip_load_weights and feeds these ids.
#
# Usage: /home/troy/diffuseR-sdxl-lib/ranvl tools/gen_fixture_sdxl_clip.R

suppressMessages(library(torch))

fixture_dir <- file.path(Sys.getenv("HOME"), ".local/share/R/diffuseR/fixtures")
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)
clipl_file <- file.path(fixture_dir, "sdxl_clip_l_weights.safetensors")
bigg_file <- file.path(fixture_dir, "sdxl_bigg_weights.safetensors")
io_file <- file.path(fixture_dir, "sdxl_clip_io.safetensors")

# int64-safe, row-major F32 safetensors writer (shared with the SDXL UNet
# fixture): the R `safetensors` writer overflows int32 when computing data
# offsets past 2 GB (corrupting the header), while yunque's reader handles
# int64 offsets fine.
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
    writeBin(length(jraw), con, size = 4L, endian = "little")
    writeBin(0L, con, size = 4L, endian = "little")
    writeBin(jraw, con)
    for (k in keys) {
        v <- as.array(tensors[[k]]$to(dtype = torch_float32())$contiguous()$reshape(c(-1L)))
        writeBin(as.numeric(v), con, size = 4L, endian = "little")
    }
}

# Refill zero/unit-init params so token+pos, LN affine, and pooled paths
# are all non-trivial (nn_linear / nn_embedding are already random-init).
randomize_flat_params <- function(model, seed) {
    torch_manual_seed(seed)
    with_no_grad({
        for (nm in names(model$named_parameters())) {
            p <- model$named_parameters()[[nm]]
            if (grepl("position_embedding$", nm)) {
                p$copy_(torch_randn_like(p) * 0.02)
            } else if (grepl("layernorm|final_layer_norm", nm) &&
                       grepl("weight$", nm)) {
                p$copy_(torch_randn_like(p) * 0.1 + 1)
            } else if (grepl("layernorm|final_layer_norm", nm) &&
                       grepl("bias$", nm)) {
                p$copy_(torch_randn_like(p) * 0.1)
            }
        }
    })
}

# Production SDXL text-encoder configs.
torch_manual_seed(100)
set.seed(100)
clipl <- diffuseR::text_encoder_native(
    vocab_size = 49408L, context_length = 77L, embed_dim = 768L,
    num_layers = 12L, num_heads = 12L, mlp_dim = 3072L,
    gelu_type = "quick", apply_final_ln = FALSE)
clipl$eval()
randomize_flat_params(clipl, 101)

torch_manual_seed(200)
set.seed(200)
bigg <- diffuseR::text_encoder2_native(
    vocab_size = 49408L, context_length = 77L, embed_dim = 1280L,
    num_layers = 32L, num_heads = 20L, mlp_dim = 5120L)
bigg$eval()
randomize_flat_params(bigg, 201)

# Fixed token-id sequence (0-based), shared by both encoders exactly like
# txt2img_sdxl.R feeds one CLIPTokenizer output to both. BOS at 1, a single
# EOS = 49407 (highest id -> unambiguous argmax) mid-sequence, random real
# tokens elsewhere; no other 49407 so which.max / torch_argmax agree.
set.seed(7)
S <- 16L
eos_pos <- 11L
ids0 <- sample.int(49405L, S, replace = TRUE)          # 1..49405 (< EOS)
ids0[1] <- 49406L                                       # BOS
ids0[eos_pos] <- 49407L                                 # EOS
ids0 <- ids0 - 1L                                       # 0-based (BOS 49405, EOS 49406)
ids_t <- torch_tensor(matrix(ids0, 1L), dtype = torch_long())  # forward adds +1

# Manual penultimate = hidden_states[-2] (state after num_layers - 1
# layers, pre-final-LN). Reproduces diffusers SDXL clip_skip = None.
penultimate <- function(model, ids_t, n_layers) {
    tok <- model$token_embedding(ids_t + 1L)
    S <- ids_t$shape[2]
    pos <- model$position_embedding[1:S, ]$unsqueeze(1)$expand(
        c(ids_t$shape[1], -1, -1))
    h <- tok + pos
    for (i in seq_len(n_layers - 1L)) {
        h <- model$transformer_blocks[[i]](h)
    }
    h
}

# bigG pooled: all layers -> final LN -> EOS (argmax) gather ->
# text_projection. Same logic as text_encoder2_native's pooled path.
bigg_pooled <- function(model, ids_t, n_layers) {
    tok <- model$token_embedding(ids_t + 1L)
    S <- ids_t$shape[2]
    pos <- model$position_embedding[1:S, ]$unsqueeze(1)$expand(
        c(ids_t$shape[1], -1, -1))
    h <- tok + pos
    for (i in seq_len(n_layers)) {
        h <- model$transformer_blocks[[i]](h)
    }
    h_ln <- model$final_layer_norm(h)
    eos <- torch_argmax(ids_t, dim = 2L, keepdim = TRUE)
    pre <- h_ln$gather(dim = 2L,
                       index = eos$unsqueeze(-1L)$expand(
                           c(-1L, -1L, model$embed_dim)))$squeeze(2L)
    model$text_projection(pre)
}

out <- with_no_grad({
    pen_l <- penultimate(clipl, ids_t, 12L)            # [1, S, 768]
    pen_g <- penultimate(bigg, ids_t, 32L)             # [1, S, 1280]
    context <- torch_cat(list(pen_l, pen_g), dim = 3L) # [1, S, 2048]
    pooled <- bigg_pooled(bigg, ids_t, 32L)            # [1, 1280]
    list(context = context, pooled = pooled)
})

# Weights (int64-safe writer).
save_st_f32(clipl$state_dict(), clipl_file)
save_st_f32(bigg$state_dict(), bigg_file)
cat(sprintf("clip-l weights: %s (%.2f GB, %d tensors)\n", clipl_file,
            file.size(clipl_file) / 1e9, length(clipl$state_dict())))
cat(sprintf("bigg   weights: %s (%.2f GB, %d tensors)\n", bigg_file,
            file.size(bigg_file) / 1e9, length(bigg$state_dict())))

# ids + reference outputs (small; the R safetensors writer is fine here).
safetensors::safe_save_file(list(
    ids0 = torch_tensor(matrix(ids0, 1L), dtype = torch_float32()),
    context = out$context$contiguous(),
    pooled = out$pooled$contiguous()
), io_file)

cat(sprintf("io: %s (%.2f MB)\n", io_file, file.size(io_file) / 1e6))
cat(sprintf("eos position (1-based which.max): %d\n", which.max(ids0)))
cat(sprintf("context %s sd %.4f range [%.3f, %.3f]\n",
            paste(dim(out$context), collapse = "x"), out$context$std()$item(),
            out$context$min()$item(), out$context$max()$item()))
cat(sprintf("pooled  %s sd %.4f range [%.3f, %.3f]\n",
            paste(dim(out$pooled), collapse = "x"), out$pooled$std()$item(),
            out$pooled$min()$item(), out$pooled$max()$item()))
