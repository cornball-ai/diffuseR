#' Z-Image-Turbo end-to-end text-to-image pipeline (anvl / XLA backend)
#'
#' anvl re-implementation of \code{diffuseR::txt2img_zimage}: Qwen3-4B
#' prompt encoding (penultimate hidden state) feeds the caption
#' conditioning, a guidance-free FlowMatch Euler loop denoises over the
#' native \code{\link{yq_zimage_dit}} at the real config, and the FLUX.1
#' 16-channel decoder (\code{\link{yq_zimage_vae_decode}}) renders pixels.
#'
#' Everything runs f32 (anvl has no fp8/bf16): the quantized transformer
#' is dequantized to f32 on load (per-tensor \code{weight_scale}), the
#' Qwen3 encoder loads bf16-upcast-to-f32, so this matches a CPU-f32
#' torch reference run (fp8 dequant + f32 compute) rather than the
#' production bf16 GPU path.
#'
#' @name anvl_zimage_pipeline
NULL

# ---- sharded fp8 checkpoint reader ---------------------------------------

# Open a sharded quantized transformer artifact (manifest.json listing
# shard files, no HF-style index.json) as a yunque "yq_sharded" object so
# yunque::st_read dispatches reads to the owning shard. Each key maps
# to the shard whose header carries it; the F8_E4M3 weights upcast to f32
# through the reader and are dequantized against their <key>_scale sibling
# by the loader below.
.yq_zimage_open_fp8 <- function(dir) {
    manifest <- jsonlite::fromJSON(file.path(dir, "manifest.json"),
                                   simplifyVector = TRUE)
    shard_paths <- file.path(dir, manifest$shards)
    sts <- lapply(shard_paths, yunque::st_open)
    names(sts) <- manifest$shards
    # unname the per-shard headers before merging: c() on a *named* list of
    # lists would prefix every key with its shard filename, which breaks the
    # key census (count_blocks) and the <key>_scale existence check (has()).
    header <- do.call(c, unname(lapply(sts, function(s) s$header)))
    key_shard <- list()
    for (s in manifest$shards) {
        for (k in names(sts[[s]]$header)) key_shard[[k]] <- s
    }
    structure(list(sts = sts, header = header, key_shard = key_shard,
                   sharded = TRUE), class = "yq_sharded")
}

.yq_zimage_close_fp8 <- function(st) {
    for (s in st$sts) close(s$con)
}

#' Load the quantized Z-Image DiT into an anvl pytree (fp8, dequantized)
#'
#' Reads the sharded fp8 transformer artifact built by
#' \code{\link{flux_quantize}} (manifest.json + F8_E4M3 shards). The 238
#' cast linears (block attention + feed-forward) are read as F8_E4M3,
#' upcast to f32 by yunque's reader, and dequantized by their per-tensor
#' \code{<key>_scale} (\code{weight_f32 = fp8 * scale}); every other
#' weight (embedders, timestep MLP, adaLN, norms, final layer, pad
#' tokens) is a BF16 upcast with no scale. The result is the same pytree
#' \code{\link{yq_zimage_dit}} expects — identical to
#' \code{\link{yq_zimage_load_weights}}, only the reader differs.
#'
#' @param dir The fp8 artifact directory (manifest.json + shards).
#' @param patch_key Character. Embedder / final-layer module-dict key.
#' @param device Character. Target device.
#'
#' @return DiT weights pytree.
#'
#' @export
yq_zimage_load_transformer_fp8 <- function(dir, patch_key = "2-1",
                                           device = "cpu") {
    st <- .yq_zimage_open_fp8(dir)
    on.exit(.yq_zimage_close_fp8(st))
    has <- function(key) !is.null(st$header[[key]])
    read_w <- function(key, transpose) {
        w <- yunque::st_read(st, key, transpose = transpose)
        sk <- paste0(key, "_scale")
        if (has(sk)) {
            w <- w * as.numeric(yunque::st_read(st, sk))
        }
        anvl::nv_array(w, dtype = "f32", device = device)
    }
    lin <- function(key) read_w(key, TRUE)
    vec <- function(key) read_w(key, FALSE)
    .yq_zimage_assemble_weights(lin, vec, names(st$header), patch_key)
}

# ---- host-side patchify / unpatchify (jit-traced with the DiT) ------------

# Latent [C, F, H, W] (F = f_tokens, pf = 1) -> packed tokens
# [1, f*h*w, pf*p*p*C], matching diffuseR::zimage_patchify.
.yq_zimage_patchify <- function(x, C, f_tokens, h_tokens, w_tokens, p) {
    v <- anvl::nv_reshape(x, c(C, f_tokens, 1L, h_tokens, p, w_tokens, p))
    v <- anvl::nv_transpose(v, c(2L, 4L, 6L, 3L, 5L, 7L, 1L))
    anvl::nv_reshape(v, c(1L, f_tokens * h_tokens * w_tokens, p * p * C))
}

# Packed final-layer output [1, S_all, p*p*C] -> latent [C, F, H, W],
# taking the first img_len (image-span) tokens; matches
# diffuseR::zimage_unpatchify.
.yq_zimage_unpatchify <- function(out, C, f_tokens, h_tokens, w_tokens, p,
                                  img_len) {
    o <- anvl::nv_reshape(out, c(anvl::shape(out)[2L], p * p * C))
    o <- o[array(seq_len(img_len)), ]
    o <- anvl::nv_reshape(o, c(f_tokens, h_tokens, w_tokens, 1L, p, p, C))
    o <- anvl::nv_transpose(o, c(7L, 1L, 4L, 2L, 5L, 3L, 6L))
    anvl::nv_reshape(o, c(C, f_tokens, h_tokens * p, w_tokens * p))
}

# ---- FlowMatch schedule (host-side) --------------------------------------

#' Z-Image-Turbo FlowMatch sigma schedule (host-side)
#'
#' The checkpoint scheduler's static exponential shift on the linear
#' sigma ramp \code{seq(1, 1/n, n)} (no dynamic shifting; Turbo's
#' calculate_shift/mu path is dead for this model). Mirrors
#' \code{flowmatch_set_timesteps(shift = 3, sigmas = seq(1, 1/n, n))}.
#' The model consumes the reversed normalized timestep
#' \code{t_model = 1 - sigma} and its velocity is negated in the Euler
#' step, so the update is \code{latents + (sigma_i - sigma_{i+1}) * v}.
#'
#' @param n_steps Integer. Denoising steps (Turbo: 8).
#' @param shift Numeric. Static schedule shift (3.0).
#'
#' @return List: \code{sigmas} (length n, shifted), \code{sigmas_full}
#'   (sigmas with the terminal 0 appended), \code{t_model} (1 - sigmas,
#'   the reversed normalized timestep fed to the DiT).
#'
#' @export
yq_zimage_sigmas <- function(n_steps, shift = 3.0) {
    base <- seq(1, 1 / n_steps, length.out = n_steps)
    sig <- shift * base / (1 + (shift - 1) * base)
    list(sigmas = sig, sigmas_full = c(sig, 0), t_model = 1 - sig)
}

# ---- end-to-end generate -------------------------------------------------

#' Generate a Z-Image-Turbo image on the anvl backend
#'
#' Runs the full text-to-image pipeline from real weights, one phase
#' resident at a time (Qwen3 encode, then DiT denoise, then VAE decode),
#' freeing each phase's weights before the next so the f32 residents do
#' not accumulate. Deterministic given the token ids and the initial
#' noise, so it can be validated against a torch reference by feeding the
#' same fixture inputs.
#'
#' @param input_ids Integer matrix \code{[1, S]} of 0-based Qwen3 token
#'   ids (chat-templated, right-padded), as produced by
#'   \code{encode_qwen(..., enable_thinking = TRUE)}.
#' @param attention_mask Integer matrix \code{[1, S]} (1 real, 0 pad).
#' @param noise AnvlArray \code{[1, 16, H/8, W/8]} initial latents.
#' @param height,width Integers, divisible by 16.
#' @param dit_dir The fp8 transformer artifact directory.
#' @param qwen_dir The Qwen3 \code{text_encoder} directory (index +
#'   shards).
#' @param vae_path Native \code{vae_decoder_native} decoder state_dict
#'   \code{.safetensors} (no \code{decoder.} prefix).
#' @param steps Integer. Denoising steps (Turbo: 8).
#' @param shift Numeric. FlowMatch static shift (3.0).
#' @param penult_layer Integer. Qwen3 layers to run; the penultimate
#'   hidden state (num_hidden_layers - 1 = 35) is the caption feature.
#' @param decode Logical. Run the VAE decode (else pixels are NULL).
#' @param device Character. Compute device.
#' @param precision Character. Matmul precision (\code{"highest"} = strict
#'   f32 for parity).
#' @param verbose Logical.
#'
#' @return List \code{list(latents, pixels, cap_feats)}: final latents
#'   \code{[1, 16, H/8, W/8]}, decoded pixels \code{[1, 3, H, W]} in
#'   [-1, 1] (or NULL), and the Qwen3 caption features
#'   \code{[1, n_real, 2560]} (for debugging the text stage).
#'
#' @export
yq_zimage_generate <- function(input_ids, attention_mask, noise,
                               height, width, dit_dir, qwen_dir, vae_path,
                               steps = 8L, shift = 3.0, penult_layer = 35L,
                               decode = TRUE, device = "cpu",
                               precision = "highest", verbose = TRUE) {
    steps <- as.integer(steps)
    input_ids <- matrix(as.integer(input_ids), nrow = 1L)
    attention_mask <- matrix(as.integer(attention_mask), nrow = 1L)
    S <- ncol(input_ids)
    n_real <- sum(attention_mask[1L, ])
    height <- as.integer(height)
    width <- as.integer(width)
    C <- 16L
    p <- 2L
    f_tokens <- 1L
    h8 <- height %/% 8L
    w8 <- width %/% 8L
    h_tokens <- h8 %/% p
    w_tokens <- w8 %/% p
    img_len <- f_tokens * h_tokens * w_tokens
    say <- function(...) if (verbose) message(...)

    ## --- Phase 1: Qwen3 encode -> caption features ---------------------------
    say("Encoding prompt (Qwen3, ", penult_layer, " layers)...")
    w_qwen <- yq_qwen3_load_weights(qwen_dir, n_layers = penult_layer,
                                    device = device)
    embeds <- yq_qwen3_embed(w_qwen$embed, input_ids, device = device)
    rope_q <- yq_qwen3_rope(S, 128L, 1e6, device = device)
    mask_q <- yq_qwen3_mask(attention_mask, S, batch = 1L, device = device)
    enc <- anvl::jit(yq_qwen3_encoder(out_layers = penult_layer,
                                      precision = precision))
    hidden <- as.array(enc(embeds, rope_q$cos, rope_q$sin, mask_q, w_qwen))
    cap_arr <- array(hidden[1L, seq_len(n_real), ], dim = c(1L, n_real, 2560L))
    cap_feats <- anvl::nv_array(cap_arr, dtype = "f32", device = device)
    rm(w_qwen, embeds, mask_q, hidden, cap_arr)
    gc(verbose = FALSE)

    ## --- Phase 2: DiT FlowMatch denoise --------------------------------------
    say("Loading transformer (fp8 -> f32)...")
    w_dit <- yq_zimage_load_transformer_fp8(dit_dir, device = device)
    rope <- yq_zimage_rope(h_tokens, w_tokens, n_real, f_tokens = f_tokens,
                           axes_dim = c(32L, 48L, 48L), theta = 256,
                           device = device)
    dit <- yq_zimage_dit(heads = 30L, precision = precision)
    step_fn <- anvl::jit(function(lat, cap, t_freq, ci, si, cc, sc, w) {
        tok <- .yq_zimage_patchify(lat, C, f_tokens, h_tokens, w_tokens, p)
        out <- dit(tok, cap, t_freq, ci, si, cc, sc, w)
        .yq_zimage_unpatchify(out, C, f_tokens, h_tokens, w_tokens, p, img_len)
    })
    sched <- yq_zimage_sigmas(steps, shift)
    lat <- anvl::nv_reshape(noise, c(C, f_tokens, h8, w8))
    say(sprintf("Denoising: %d steps at %dx%d...", steps, width, height))
    for (i in seq_len(steps)) {
        t_freq <- yq_zimage_time_embed(sched$t_model[i], device = device)
        vel <- step_fn(lat, cap_feats, t_freq, rope$cos_img, rope$sin_img,
                       rope$cos_cap, rope$sin_cap, w_dit)
        d <- sched$sigmas_full[i] - sched$sigmas_full[i + 1L]   # = -(sigma_next - sigma)
        lat <- lat + vel * anvl::nv_scalar(d, "f32", device = device)
    }
    latents <- anvl::nv_reshape(lat, c(1L, C, h8, w8))
    anvl::await(latents)
    rm(w_dit, step_fn, lat)
    gc(verbose = FALSE)

    ## --- Phase 3: VAE decode -------------------------------------------------
    pixels <- NULL
    if (decode) {
        say("Decoding (16-channel VAE)...")
        w_vae <- yq_zimage_vae_load_weights(vae_path, device = device)
        dec <- anvl::jit(function(z) yq_zimage_vae_decode(z, w_vae))
        pixels <- dec(yq_zimage_vae_prepare(latents))
        anvl::await(pixels)
        rm(w_vae, dec)
        gc(verbose = FALSE)
    }

    list(latents = latents, pixels = pixels, cap_feats = cap_feats)
}
