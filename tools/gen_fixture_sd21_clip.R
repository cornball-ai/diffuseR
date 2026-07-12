# Generate the SD 2.1 CLIP text-encoder parity fixture (torch reference).
# Loads the real F16 diffusers text_encoder checkpoint into
# diffuseR::text_encoder_native (upcast to f32) via the SAME call the
# pipeline uses (apply_final_ln = TRUE, default gelu_type = "tanh"), runs
# it on a fixed small random token-id sequence, and saves the ids + output
# hidden states (the final-LN last hidden state SD 2.1 consumes) to an f32
# safetensors fixture. The anvl test reloads the same checkpoint via
# yq_clip_load_weights and feeds these ids.
#
# Usage: /home/troy/diffuseR-anvl-lib/ranvl tools/gen_fixture_sd21_clip.R

suppressMessages(library(torch))

te <- file.path(Sys.getenv("HOME"),
                ".local/share/R/diffuseR/sd21-diffusers/text_encoder")
stopifnot(dir.exists(te))

fixture_dir <- file.path(Sys.getenv("HOME"), ".local/share/R/diffuseR/fixtures")
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)
fixture <- file.path(fixture_dir, "sd21_clip.safetensors")

# Same construction the SD 2.1 pipeline uses: final-LN last hidden state,
# tanh-approximation GELU (the config carries no hidden_act).
enc <- diffuseR::text_encoder_native_from_safetensors(te, apply_final_ln = TRUE,
                                                      verbose = TRUE)
enc$eval()

set.seed(21)
torch_manual_seed(21)
S <- 16L
vocab <- 49408L
ids0 <- sample.int(vocab, S, replace = TRUE) - 1L        # 0-based token ids
ids_t <- torch_tensor(matrix(ids0, 1L), dtype = torch_long())  # forward adds +1

out <- with_no_grad(enc(ids_t))                          # [1, S, 1024]

safetensors::safe_save_file(list(
    ids0 = torch_tensor(matrix(ids0, 1L), dtype = torch_float32()),
    out = out$contiguous()
), fixture)

cat(sprintf("fixture: %s (%.2f MB)\n", fixture, file.size(fixture) / 1e6))
cat(sprintf("out shape %s sd %.4f range [%.3f, %.3f]\n",
            paste(dim(out), collapse = "x"), out$std()$item(),
            out$min()$item(), out$max()$item()))
