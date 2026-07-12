# Z-Image-Turbo end-to-end parity fixture (torch reference, REAL weights).
#
# Runs the production diffuseR::txt2img_zimage path component-by-component
# on CPU f32 (fp8 transformer dequantized to f32, bf16 Qwen3 upcast to
# f32) so the anvl port can be validated apples-to-apples. Captures the
# exact inputs the anvl pipeline consumes (initial noise, 0-based token
# ids, attention mask) plus the intermediate caption conditioning and the
# two outputs to compare (final latents, decoded pixels). Also writes the
# VAE decoder's native state_dict so the anvl loader can read the same
# weights.
#
# The fixture bundle is written with a small int64-safe row-major F32
# safetensors writer (safetensors::safe_save_file overflows its int32
# offsets past ~2 GB); here the tensors are tiny, but the writer is the
# right tool for large-model fixtures and is round-trip-checked below.
#
# Usage: /home/troy/diffuseR-zimage-lib/ranvl tools/gen_fixture_zimage_e2e.R

suppressMessages({
    library(torch)
    library(diffuseR)
})

## ---- fixture config (MUST match inst/tinytest/anvl_test_zimage_e2e.R) ----
prompt <- "a red cube on a wooden table"
seed <- 42L
height <- 128L
width <- 128L
steps <- 4L
max_seq <- 64L
shift <- 3.0

fixture_dir <- file.path(Sys.getenv("HOME"), ".local/share/R/diffuseR/fixtures")
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)
fixture_path <- file.path(fixture_dir, "zimage_e2e.safetensors")
vae_weights_path <- file.path(fixture_dir, "zimage_e2e_vae.safetensors")

## ---- int64-safe, row-major F32 safetensors writer -----------------------
# Header offsets are doubles (int64-safe); the 8-byte header length is
# written as 4 low bytes + 4 zero bytes (little-endian). Tensor bytes are
# row-major f32, matching what anvl::nv_read expects.
st_write_f32 <- function(tensors, path) {
    parts <- character(0)
    blobs <- vector("list", length(tensors))
    names(blobs) <- names(tensors)
    offset <- 0
    for (nm in names(tensors)) {
        t <- tensors[[nm]]$to(dtype = torch_float32())$contiguous()$cpu()
        shp <- as.integer(t$shape)
        arr <- as.array(t)
        flat <- if (length(shp) > 1L) {
            as.double(aperm(arr, rev(seq_along(shp))))   # -> row-major
        } else {
            as.double(arr)
        }
        raw <- writeBin(flat, raw(), size = 4L, endian = "little")
        blobs[[nm]] <- raw
        shape_str <- if (length(shp) == 0L) {
            "[]"
        } else {
            paste0("[", paste(shp, collapse = ","), "]")
        }
        parts <- c(parts, sprintf(
            '"%s":{"dtype":"F32","shape":%s,"data_offsets":[%s,%s]}',
            nm, shape_str,
            format(offset, scientific = FALSE, trim = TRUE),
            format(offset + length(raw), scientific = FALSE, trim = TRUE)))
        offset <- offset + length(raw)
    }
    hb <- charToRaw(paste0("{", paste(parts, collapse = ","), "}"))
    con <- file(path, "wb")
    on.exit(close(con))
    writeBin(length(hb), con, size = 4L, endian = "little")
    writeBin(0L, con, size = 4L, endian = "little")
    writeBin(hb, con)
    for (nm in names(blobs)) writeBin(blobs[[nm]], con)
}

## ---- load the production pipeline on CPU f32 (fp8 dequant) ---------------
message("Loading Z-Image-Turbo pipeline (CPU f32, fp8 dequant)...")
pipe <- zimage_load_pipeline(device = "cpu", precision = "fp8",
                             phase_offload = FALSE, verbose = TRUE)

## ---- Phase 1: Qwen3 encode (capture ids + caption features) -------------
message("Encoding prompt (Qwen3)...")
enc <- encode_qwen(pipe$tokenizer, prompt, max_length = max_seq,
                   chat_template = TRUE, enable_thinking = TRUE)
long <- torch_long()
ids_1based <- torch_tensor(enc$input_ids + 1L, dtype = long)   # embed is 1-based
mask_t <- torch_tensor(enc$attention_mask, dtype = long)
states <- with_no_grad(pipe$text_encoder(ids_1based, attention_mask = mask_t,
                                         out_layers = pipe$te_penult_layer))
n_real <- sum(enc$attention_mask[1, ])
cap_feats <- states[[1]][1, 1:n_real, ]$contiguous()            # [n_real, 2560]
message(sprintf("  S=%d  n_real=%d  penult_layer=%d  cap %s",
                max_seq, n_real, pipe$te_penult_layer,
                paste(dim(cap_feats), collapse = "x")))

## ---- Phase 2: noise + FlowMatch denoise ---------------------------------
h8 <- height %/% 8L
w8 <- width %/% 8L
torch_manual_seed(seed)
noise <- torch_randn(c(1L, 16L, h8, w8), dtype = torch_float32())

sched <- flowmatch_scheduler_create(shift = shift, use_dynamic_shifting = FALSE)
sched <- flowmatch_set_timesteps(sched, steps,
                                 sigmas = seq(1, 1 / steps, length.out = steps))
message(sprintf("Denoising: %d steps at %dx%d...", steps, width, height))
final_latents <- diffuseR:::.zimage_denoise(
    pipe$transformer, noise$clone(), sched, cap_feats,
    torch_float32(), chunk_size = NULL, verbose = TRUE)          # [1, 16, h8, w8]

## ---- Phase 3: VAE decode ------------------------------------------------
message("Decoding...")
z_in <- final_latents$div(pipe$vae_scaling_factor)$add(pipe$vae_shift_factor)
pixels <- with_no_grad(pipe$decoder(z_in))                       # [1, 3, H, W]

## ---- save the VAE decoder native state_dict (anvl loader reads it) -------
sd <- lapply(pipe$decoder$state_dict(), function(t) t$detach()$contiguous())
safetensors::safe_save_file(sd, vae_weights_path)
message(sprintf("VAE weights: %s (%.1f MB, %d tensors)",
                vae_weights_path, file.size(vae_weights_path) / 1e6, length(sd)))

## ---- write the fixture bundle -------------------------------------------
fixture <- list(
    noise = noise$contiguous(),                                  # [1, 16, h8, w8]
    input_ids = torch_tensor(enc$input_ids, dtype = torch_float32()),   # 0-based
    attention_mask = torch_tensor(enc$attention_mask, dtype = torch_float32()),
    cap_feats = cap_feats,                                        # [n_real, 2560]
    latents = final_latents$contiguous(),                        # [1, 16, h8, w8]
    pixels = pixels$contiguous()                                 # [1, 3, H, W]
)
st_write_f32(fixture, fixture_path)

## ---- round-trip self-check of the int64-safe writer ---------------------
if (requireNamespace("anvl", quietly = TRUE)) {
    rt <- anvl::nv_read(fixture_path)
    d_noise <- max(abs(as.array(rt$noise) - as.array(noise)))
    d_pix <- max(abs(as.array(rt$pixels) - as.array(pixels)))
    message(sprintf("writer round-trip: noise %.2e  pixels %.2e", d_noise, d_pix))
    stopifnot(d_noise < 1e-6, d_pix < 1e-5)
}

message(sprintf("fixture: %s (%.2f MB)", fixture_path,
                file.size(fixture_path) / 1e6))
message(sprintf("latents sd %.4f range [%.3f, %.3f]",
                final_latents$std()$item(), final_latents$min()$item(),
                final_latents$max()$item()))
message(sprintf("pixels sd %.4f range [%.3f, %.3f]",
                pixels$std()$item(), pixels$min()$item(), pixels$max()$item()))
