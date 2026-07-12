#' Z-Image DiT transformer (anvl port)
#'
#' anvl re-implementation of \code{diffuseR::zimage_transformer}
#' (ZImageTransformer2DModel): a single-stream DiT where patchified image
#' tokens pass through a timestep-modulated noise refiner, caption tokens
#' through an unmodulated context refiner, then both are concatenated
#' (image first) and run through the main trunk before an adaLN final
#' layer projects back to patch space.
#'
#' Every block uses sandwich RMS norms (a learned norm before AND after
#' both attention and feed-forward) and a scale/gate-only modulation:
#' four chunks (scale_msa, gate_msa, scale_mlp, gate_mlp), no shift, gates
#' \code{tanh}-squashed, scales \code{1 + x}. Attention is plain joint
#' self-attention with per-head RMS q/k norms and 3-axis interleaved RoPE
#' (theta 256). The context refiner blocks carry no adaLN weights.
#'
#' Returns a closure over the static config; \code{anvl::jit()} it. The
#' timestep sinusoid (\code{\link{yq_zimage_time_embed}}) and RoPE tables
#' (\code{\link{yq_zimage_rope}}) are precomputed host-side and passed as
#' inputs; the image patchify happens host-side too (the closure consumes
#' patchified tokens and returns the packed final-layer output, matching
#' the FLUX.2 boundary). Batch-of-1.
#'
#' @param heads Integer. Attention heads (full model: 30). Head dim is
#'   derived as \code{dim / heads}.
#' @param eps Numeric. RMSNorm epsilon for the blocks (1e-5).
#' @param final_eps Numeric. LayerNorm epsilon for the final layer (1e-6).
#' @param precision Character. Matmul precision (see
#'   \code{\link[yunque]{yq_linear}}).
#'
#' @return Function of (tokens, cap_feats, t_freq, cos_img, sin_img,
#'   cos_cap, sin_cap, w):
#'   \itemize{
#'     \item tokens \code{[1, img_len, patch_dim]} patchified latent
#'     \item cap_feats \code{[1, cap_len, cap_feat_dim]} caption features
#'     \item t_freq \code{[1, 256]} timestep sinusoid
#'       (\code{\link{yq_zimage_time_embed}})
#'     \item cos_img, sin_img \code{[img_padded, head_dim]} image RoPE
#'     \item cos_cap, sin_cap \code{[cap_padded, head_dim]} caption RoPE
#'     \item w weights pytree (\code{\link{yq_zimage_load_weights}})
#'   }
#'   returning the packed final-layer output \code{[1, S_all, patch_dim]}.
#'
#' @export
yq_zimage_dit <- function(heads = 30L, eps = 1e-5, final_eps = 1e-6,
                          precision = "highest") {
    heads <- as.integer(heads)

    lin <- function(x, w_t, bias = NULL) {
        yunque::yq_linear(x, w_t, bias = bias, precision = precision)
    }

    attention <- function(x, cos, sin, wb) {
        s <- anvl::shape(x)
        b <- s[1L]; n <- s[2L]; dim <- s[3L]
        head_dim <- dim %/% heads
        inner <- heads * head_dim

        q <- lin(x, wb$to_q)
        k <- lin(x, wb$to_k)
        v <- lin(x, wb$to_v)
        q <- anvl::nv_reshape(q, c(b, n, heads, head_dim))
        k <- anvl::nv_reshape(k, c(b, n, heads, head_dim))
        v <- anvl::nv_reshape(v, c(b, n, heads, head_dim))
        q <- yunque::yq_rms_norm(q, wb$norm_q, eps = eps)
        k <- yunque::yq_rms_norm(k, wb$norm_k, eps = eps)

        perm <- c(1L, 3L, 2L, 4L)              # [B, S, H, D] -> [B, H, S, D]
        q <- anvl::nv_transpose(q, perm)
        k <- anvl::nv_transpose(k, perm)
        v <- anvl::nv_transpose(v, perm)

        hs <- c(b, heads, n, head_dim)
        cs <- anvl::nv_broadcast_to(cos, hs)
        sn <- anvl::nv_broadcast_to(sin, hs)
        q <- yunque::yq_rope_apply(q, cs, sn)
        k <- yunque::yq_rope_apply(k, cs, sn)

        attn <- yunque::yq_sdpa(q, k, v, precision = precision)
        attn <- anvl::nv_reshape(anvl::nv_transpose(attn, perm), c(b, n, inner))
        lin(attn, wb$to_out)
    }

    feed_forward <- function(x, wb) {
        lin(yunque::yq_silu(lin(x, wb$w1)) * lin(x, wb$w3), wb$w2)
    }

    block <- function(x, cos, sin, adaln, wb, modulation) {
        s <- anvl::shape(x)
        dim <- s[3L]
        if (modulation) {
            b <- anvl::shape(adaln)[1L]
            mod <- anvl::nv_reshape(lin(adaln, wb$adaLN, wb$adaLN_b),
                                    c(b, 1L, 4L * dim))
            scale_msa <- yunque::yq_slice_lastdim(mod, 1L, dim) + 1
            gate_msa <- anvl::nv_tanh(yunque::yq_slice_lastdim(mod, dim + 1L, 2L * dim))
            scale_mlp <- yunque::yq_slice_lastdim(mod, 2L * dim + 1L, 3L * dim) + 1
            gate_mlp <- anvl::nv_tanh(yunque::yq_slice_lastdim(mod, 3L * dim + 1L,
                                                               4L * dim))

            n1 <- yunque::yq_rms_norm(x, wb$attn_norm1, eps = eps) *
                anvl::nv_broadcast_to(scale_msa, s)
            attn <- attention(n1, cos, sin, wb)
            x <- x + anvl::nv_broadcast_to(gate_msa, s) *
                yunque::yq_rms_norm(attn, wb$attn_norm2, eps = eps)

            n2 <- yunque::yq_rms_norm(x, wb$ffn_norm1, eps = eps) *
                anvl::nv_broadcast_to(scale_mlp, s)
            ff <- feed_forward(n2, wb)
            x + anvl::nv_broadcast_to(gate_mlp, s) *
                yunque::yq_rms_norm(ff, wb$ffn_norm2, eps = eps)
        } else {
            n1 <- yunque::yq_rms_norm(x, wb$attn_norm1, eps = eps)
            attn <- attention(n1, cos, sin, wb)
            x <- x + yunque::yq_rms_norm(attn, wb$attn_norm2, eps = eps)
            n2 <- yunque::yq_rms_norm(x, wb$ffn_norm1, eps = eps)
            ff <- feed_forward(n2, wb)
            x + yunque::yq_rms_norm(ff, wb$ffn_norm2, eps = eps)
        }
    }

    pad_rows <- function(tok, dim, pad) {
        rows <- anvl::nv_reshape(tok, c(1L, 1L, dim))
        anvl::nv_broadcast_to(rows, c(1L, pad, dim))
    }

    function(tokens, cap_feats, t_freq, cos_img, sin_img, cos_cap, sin_cap, w) {
        # Timestep -> adaLN conditioning: MLP over the host-side sinusoid.
        adaln <- lin(yunque::yq_silu(lin(t_freq, w$t_mlp0, w$t_mlp0_b)),
                     w$t_mlp2, w$t_mlp2_b)

        # Image tokens: embed, pad to a multiple of 32, refine.
        x <- lin(tokens, w$x_embed, w$x_embed_b)
        dim <- anvl::shape(x)[3L]
        img_len <- anvl::shape(x)[2L]
        img_pad <- (-img_len) %% 32L
        if (img_pad > 0L) {
            x <- anvl::nv_concatenate(x, pad_rows(w$x_pad_token, dim, img_pad),
                                      dimension = 2L)
        }
        for (i in seq_along(w$noise_refiner)) {
            x <- block(x, cos_img, sin_img, adaln, w$noise_refiner[[i]], TRUE)
        }

        # Caption tokens: RMS-norm + embed, pad, refine (no modulation).
        cap <- yunque::yq_rms_norm(cap_feats, w$cap_norm, eps = eps)
        cap <- lin(cap, w$cap_lin, w$cap_lin_b)
        cap_len <- anvl::shape(cap)[2L]
        cap_pad <- (-cap_len) %% 32L
        if (cap_pad > 0L) {
            cap <- anvl::nv_concatenate(cap, pad_rows(w$cap_pad_token, dim, cap_pad),
                                        dimension = 2L)
        }
        for (i in seq_along(w$context_refiner)) {
            cap <- block(cap, cos_cap, sin_cap, NULL, w$context_refiner[[i]], FALSE)
        }

        # Unified sequence, image first.
        unified <- anvl::nv_concatenate(x, cap, dimension = 2L)
        cos_uni <- anvl::nv_concatenate(cos_img, cos_cap, dimension = 1L)
        sin_uni <- anvl::nv_concatenate(sin_img, sin_cap, dimension = 1L)
        for (i in seq_along(w$layers)) {
            unified <- block(unified, cos_uni, sin_uni, adaln, w$layers[[i]], TRUE)
        }

        # adaLN final layer: scale-only modulation (silu then linear, +1).
        su <- anvl::shape(unified)
        scale <- anvl::nv_reshape(lin(yunque::yq_silu(adaln), w$final_adaLN,
                                      w$final_adaLN_b) + 1,
                                  c(su[1L], 1L, dim))
        normed <- yunque::yq_layer_norm(unified, eps = final_eps) *
            anvl::nv_broadcast_to(scale, su)
        lin(normed, w$final_lin, w$final_lin_b)
    }
}

#' Z-Image timestep sinusoid (host-side)
#'
#' Parameter-free sinusoidal embedding of \code{t * t_scale}, cos-first
#' (\code{flip_sin_to_cos = TRUE}, \code{downscale_freq_shift = 0}),
#' matching \code{diffuseR::ltx23_get_timestep_embedding} as called by the
#' Z-Image \code{t_embedder}. Computed in base R and returned as an
#' \code{AnvlArray}, mirroring how the RoPE tables are precomputed outside
#' the model.
#'
#' @param t Numeric vector of timesteps in \code{[0, 1]}.
#' @param freq_size Integer. Sinusoid width (256).
#' @param t_scale Numeric. Timestep scale (1000).
#' @param max_period Numeric. Base period (10000).
#' @param device Character. Target device.
#'
#' @return AnvlArray \code{[length(t), freq_size]}, f32.
#'
#' @export
yq_zimage_time_embed <- function(t, freq_size = 256L, t_scale = 1000,
                                 max_period = 10000, device = "cpu") {
    ts <- as.numeric(t) * t_scale
    half <- freq_size %/% 2L
    exponent <- -log(max_period) * (0:(half - 1L)) / half
    freq <- exp(exponent)
    ang <- outer(ts, freq)
    emb <- cbind(cos(ang), sin(ang))          # cos-first
    anvl::nv_array(emb, dtype = "f32", device = device)
}

# Round doubles to f32 precision (mirrors the reference .float() cast on
# the RoPE angle before torch.polar).
.zimage_f32 <- function(x) {
    readBin(writeBin(as.double(x), raw(), size = 4L), "double",
            n = length(x), size = 4L)
}

# 3-axis interleaved rotary tables for one set of position ids. Angles
# are cast to f32 before cos/sin (reference: outer(...).float()), then
# each angle doubled by interleave (repeat_interleave(2)).
.zimage_rope_tables <- function(ids, axes_dim, theta, device) {
    dbl <- function(m) m[, rep(seq_len(ncol(m)), each = 2L), drop = FALSE]
    n_axes <- ncol(ids)
    cos_parts <- vector("list", n_axes)
    sin_parts <- vector("list", n_axes)
    for (i in seq_len(n_axes)) {
        d <- axes_dim[i]
        exponents <- seq(0, d - 2L, by = 2L)
        freqs <- 1 / theta^(exponents / d)
        ang <- outer(ids[, i], freqs)
        ang <- matrix(.zimage_f32(ang), nrow = nrow(ang))
        cos_parts[[i]] <- dbl(cos(ang))
        sin_parts[[i]] <- dbl(sin(ang))
    }
    list(cos = anvl::nv_array(do.call(cbind, cos_parts), dtype = "f32",
                              device = device),
         sin = anvl::nv_array(do.call(cbind, sin_parts), dtype = "f32",
                              device = device))
}

#' Z-Image RoPE tables for a txt2img forward (host-side)
#'
#' Base-R port of the Z-Image position scheme
#' (\code{diffuseR::zimage_cap_pos_ids} / \code{zimage_img_pos_ids} +
#' \code{zimage_pos_embed}). Caption tokens ramp \code{1..cap_padded} on
#' axis 1; image tokens carry the frame index (offset just past the
#' caption) on axis 1 and the token row/column on axes 2/3. Each
#' sub-sequence is padded to a multiple of 32. Returns separate image and
#' caption tables (the main trunk concatenates them, image first).
#'
#' @param h_tokens Integer. Token grid height (latent height / patch).
#' @param w_tokens Integer. Token grid width (latent width / patch).
#' @param cap_len Integer. Caption length before padding.
#' @param f_tokens Integer. Token grid frames (1 for txt2img).
#' @param axes_dim Integer vector of per-axis rotary dims (must sum to the
#'   attention head dim; full model: \code{c(32, 48, 48)}).
#' @param theta Numeric. Base frequency (256).
#' @param seq_multi Integer. Sub-sequence padding multiple (32).
#' @param device Character. Target device.
#'
#' @return List \code{list(cos_img, sin_img, cos_cap, sin_cap)}, each an
#'   \code{AnvlArray} \code{[padded_len, sum(axes_dim)]}, f32.
#'
#' @export
yq_zimage_rope <- function(h_tokens, w_tokens, cap_len, f_tokens = 1L,
                           axes_dim = c(32L, 48L, 48L), theta = 256,
                           seq_multi = 32L, device = "cpu") {
    pad_len <- function(n) (-n) %% seq_multi
    cap_padded <- cap_len + pad_len(cap_len)

    cap_ids <- matrix(0, nrow = cap_padded, ncol = 3L)
    cap_ids[, 1L] <- seq_len(cap_padded)      # 1..cap_padded

    start0 <- cap_padded + 1L
    n_img <- f_tokens * h_tokens * w_tokens
    fi <- rep(seq_len(f_tokens) - 1L + start0, each = h_tokens * w_tokens)
    hi <- rep(rep(seq_len(h_tokens) - 1L, each = w_tokens), times = f_tokens)
    wi <- rep(seq_len(w_tokens) - 1L, times = f_tokens * h_tokens)
    img_ids <- cbind(fi, hi, wi)
    img_pad <- pad_len(n_img)
    if (img_pad > 0L) {
        img_ids <- rbind(img_ids, matrix(0, img_pad, 3L))
    }

    cap_t <- .zimage_rope_tables(cap_ids, axes_dim, theta, device)
    img_t <- .zimage_rope_tables(img_ids, axes_dim, theta, device)
    list(cos_img = img_t$cos, sin_img = img_t$sin,
         cos_cap = cap_t$cos, sin_cap = cap_t$sin)
}

#' Load Z-Image DiT weights into an anvl pytree
#'
#' Reads a \code{zimage_transformer} state-dict safetensors file (f32),
#' transposing 2-D linears to \code{[in, out]} and wrapping each tensor as
#' an \code{AnvlArray} on \code{device}. Block counts are derived from the
#' key census (\code{layers.N}, \code{noise_refiner.N},
#' \code{context_refiner.N}). Returns the nested list
#' \code{\link{yq_zimage_dit}} expects. The patch key is \code{"2-1"}
#' (patch_size 2, f_patch_size 1).
#'
#' @param path Path to the state-dict safetensors file.
#' @param patch_key Character. Embedder / final-layer module-dict key.
#' @param device Character. Target device.
#'
#' @return Weights pytree.
#'
#' @export
yq_zimage_load_weights <- function(path, patch_key = "2-1", device = "cpu") {
    st <- yunque::yq_st_open(path)
    on.exit(close(st$con))
    lin <- function(key) anvl::nv_array(yunque::yq_st_read(st, key, transpose = TRUE),
                                        dtype = "f32", device = device)
    vec <- function(key) anvl::nv_array(yunque::yq_st_read(st, key),
                                        dtype = "f32", device = device)
    .yq_zimage_assemble_weights(lin, vec, names(st$header), patch_key)
}

# Assemble the DiT pytree from a checkpoint's key census and two readers:
# lin() reads a 2-D linear transposed to [in, out]; vec() reads a raw
# vector/scalar. Shared by the single-file f32 loader
# (\code{yq_zimage_load_weights}) and the sharded fp8 loader
# (\code{yq_zimage_load_transformer_fp8}), which differ only in how those
# two readers open shards and apply per-tensor dequant scales.
.yq_zimage_assemble_weights <- function(lin, vec, keys, patch_key) {
    count_blocks <- function(prefix) {
        m <- regmatches(keys, regexpr(paste0("^", prefix, "\\.[0-9]+\\."), keys))
        idx <- as.integer(sub(paste0("^", prefix, "\\.([0-9]+)\\.$"), "\\1", m))
        if (length(idx) == 0L) 0L else max(idx) + 1L
    }

    block_w <- function(prefix, modulation) {
        p <- paste0(prefix, ".")
        b <- list(
            norm_q = vec(paste0(p, "attention.norm_q.weight")),
            norm_k = vec(paste0(p, "attention.norm_k.weight")),
            to_q = lin(paste0(p, "attention.to_q.weight")),
            to_k = lin(paste0(p, "attention.to_k.weight")),
            to_v = lin(paste0(p, "attention.to_v.weight")),
            to_out = lin(paste0(p, "attention.to_out.0.weight")),
            w1 = lin(paste0(p, "feed_forward.w1.weight")),
            w2 = lin(paste0(p, "feed_forward.w2.weight")),
            w3 = lin(paste0(p, "feed_forward.w3.weight")),
            attn_norm1 = vec(paste0(p, "attention_norm1.weight")),
            ffn_norm1 = vec(paste0(p, "ffn_norm1.weight")),
            attn_norm2 = vec(paste0(p, "attention_norm2.weight")),
            ffn_norm2 = vec(paste0(p, "ffn_norm2.weight"))
        )
        if (modulation) {
            b$adaLN <- lin(paste0(p, "adaLN_modulation.0.weight"))
            b$adaLN_b <- vec(paste0(p, "adaLN_modulation.0.bias"))
        }
        b
    }

    xk <- paste0("all_x_embedder.", patch_key, ".")
    fk <- paste0("all_final_layer.", patch_key, ".")
    w <- list(
        x_embed = lin(paste0(xk, "weight")),
        x_embed_b = vec(paste0(xk, "bias")),
        x_pad_token = vec("x_pad_token"),
        cap_pad_token = vec("cap_pad_token"),
        cap_norm = vec("cap_embedder.0.weight"),
        cap_lin = lin("cap_embedder.1.weight"),
        cap_lin_b = vec("cap_embedder.1.bias"),
        t_mlp0 = lin("t_embedder.mlp.0.weight"),
        t_mlp0_b = vec("t_embedder.mlp.0.bias"),
        t_mlp2 = lin("t_embedder.mlp.2.weight"),
        t_mlp2_b = vec("t_embedder.mlp.2.bias"),
        final_adaLN = lin(paste0(fk, "adaLN_modulation.1.weight")),
        final_adaLN_b = vec(paste0(fk, "adaLN_modulation.1.bias")),
        final_lin = lin(paste0(fk, "linear.weight")),
        final_lin_b = vec(paste0(fk, "linear.bias"))
    )

    n_noise <- count_blocks("noise_refiner")
    n_context <- count_blocks("context_refiner")
    n_layers <- count_blocks("layers")
    w$noise_refiner <- lapply(seq_len(n_noise) - 1L,
                              function(i) block_w(sprintf("noise_refiner.%d", i), TRUE))
    w$context_refiner <- lapply(seq_len(n_context) - 1L,
                                function(i) block_w(sprintf("context_refiner.%d", i), FALSE))
    w$layers <- lapply(seq_len(n_layers) - 1L,
                       function(i) block_w(sprintf("layers.%d", i), TRUE))
    w
}
