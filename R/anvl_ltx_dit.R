#' LTX-2.3 audio-video DiT transformer (anvl port)
#'
#' anvl re-implementation of \code{diffuseR::ltx23_transformer}
#' (LTX-2.3 dual-stream audio/video DiT). Video and audio latent tokens
#' each run a stream of: modulated RMS-normed self-attention (per-head
#' output gating, split RoPE), text cross-attention (with LTX-2.3
#' query-side and key/value-side modulation and gating), optional
#' audio<->video cross-attention (a2v / v2a, with global + per-block
#' modulation and temporal-only cross RoPE), and a modulated GELU-tanh
#' feed-forward. Per-token timestep modulation feeds every block; a global
#' sigma drives the prompt (text) modulation; the output layer applies a
#' per-token scale/shift from the embedded timestep.
#'
#' Field names mirror the diffusers / \code{dit_ltx23} module tree so
#' checkpoint keys map 1:1. Weights travel as a nested named list
#' (pytree); all 2-D linears are pre-transposed to \code{[in, out]}. The
#' parameter-free timestep sinusoids and RoPE cos/sin tables are
#' precomputed host-side (\code{\link{yq_ltx_time_sinusoid}} and the
#' reference RoPE embedder) and passed as inputs, matching the FLUX.2 /
#' Z-Image boundary.
#'
#' @name anvl_ltx_dit
NULL

# --- GELU (tanh approximation), matching nnf_gelu(approximate="tanh") ---
.yq_ltx_gelu_tanh <- function(x) {
    c0 <- sqrt(2 / pi) # 0.7978845608...
    inner <- (x + (x * x * x) * 0.044715) * c0
    (x * 0.5) * (anvl::nv_tanh(inner) + 1)
}

# Static row p of a 2-D table [n, dim] as a [1, dim] AnvlArray.
.yq_ltx_row <- function(table, p) {
    s <- anvl::shape(table)
    anvl::nv_static_slice(table, start_indices = c(as.integer(p), 1L),
                          limit_indices = c(as.integer(p), s[2L]),
                          strides = c(1L, 1L))
}

# Static contiguous rows [from, to] of a 2-D table.
.yq_ltx_rows <- function(table, from, to) {
    s <- anvl::shape(table)
    anvl::nv_static_slice(table, start_indices = c(as.integer(from), 1L),
                          limit_indices = c(as.integer(to), s[2L]),
                          strides = c(1L, 1L))
}

# ltx23_get_mod_params: table[None,None] + temb.reshape(B, V, num, dim),
# unbound along the parameter axis. temb is [B, V, num*dim] (param p in
# columns [(p-1)*dim, p*dim)); table is [num, dim]. Returns a list of
# num tensors, each [B, V, dim].
.yq_ltx_mod <- function(temb, table, num, dim) {
    s <- anvl::shape(temb) ; b <- s[1L]; v <- s[2L]
    lapply(seq_len(num), function(p) {
        chunk <- yunque::yq_slice_lastdim(temb, (p - 1L) * dim + 1L, p * dim)
        row <- anvl::nv_broadcast_to(anvl::nv_reshape(.yq_ltx_row(table, p),
                c(1L, 1L, dim)),
                                     c(b, v, dim))
        chunk + row
    })
}

# adaLN-single: embedded_timestep = linear_2(silu(linear_1(sinusoid)));
# mod = linear(silu(embedded_timestep)). sinusoid is [N, 256]; returns
# list(mod [N, num_mod*dim], emb [N, dim]).
.yq_ltx_adaln <- function(sinusoid, w, precision) {
    e <- yunque::yq_linear(
                           yunque::yq_silu(yunque::yq_linear(sinusoid, w$l1, bias = w$l1b,
                precision = precision)),
                           w$l2, bias = w$l2b, precision = precision)
    mod <- yunque::yq_linear(yunque::yq_silu(e), w$lin, bias = w$linb,
                             precision = precision)
    list(mod = mod, emb = e)
}

# Apply split RoPE to a flat [B, S, inner] tensor grouped into rope_heads
# heads (the reference reshapes to the cos/sin head count, rotates, then
# flattens back). cos/sin are [B, rope_heads, S, inner/(2*rope_heads)].
.yq_ltx_rope_flat <- function(x, cos, sin, rope_heads) {
    s <- anvl::shape(x) ; b <- s[1L]; n <- s[2L]; inner <- s[3L]
    d <- inner %/% rope_heads
    perm <- c(1L, 3L, 2L, 4L)
    xh <- anvl::nv_transpose(anvl::nv_reshape(x, c(b, n, rope_heads, d)), perm)
    xh <- yunque::yq_rope_split(xh, cos, sin)
    anvl::nv_reshape(anvl::nv_transpose(xh, perm), c(b, n, inner))
}

# LTX attention closure. norm_q/norm_k RMS-normalize the whole inner_dim
# (before the head split); RoPE (when supplied) is applied flat with a
# possibly-different head grouping than attention; optional per-head
# sigmoid output gating (factor 2). qin drives the gate logits.
.yq_ltx_attn <- function(heads, head_dim, gated, precision, eps = 1e-6) {
    inner <- heads * head_dim
    perm <- c(1L, 3L, 2L, 4L)
    function(qin, kin, w, cos_q = NULL, sin_q = NULL, cos_k = NULL, sin_k = NULL,
             mask = NULL, rope_heads_q = heads, rope_heads_k = heads) {
        b <- anvl::shape(qin)[1L]
        nq <- anvl::shape(qin)[2L]
        nk <- anvl::shape(kin)[2L]

        gate_logits <- if (gated) {
            yunque::yq_linear(qin, w$gate, bias = w$gate_b,
                              precision = precision)
        }
        v <- yunque::yq_linear(kin, w$to_v, bias = w$to_v_b, precision = precision)
        q <- yunque::yq_rms_norm(
                                 yunque::yq_linear(qin, w$to_q, bias = w$to_q_b, precision = precision),
                                 w$norm_q, eps = eps)
        k <- yunque::yq_rms_norm(
                                 yunque::yq_linear(kin, w$to_k, bias = w$to_k_b, precision = precision),
                                 w$norm_k, eps = eps)
        if (!is.null(cos_q)) {
            q <- .yq_ltx_rope_flat(q, cos_q, sin_q, rope_heads_q)
            k <- .yq_ltx_rope_flat(k, cos_k, sin_k, rope_heads_k)
        }
        qh <- anvl::nv_transpose(anvl::nv_reshape(q, c(b, nq, heads, head_dim)), perm)
        kh <- anvl::nv_transpose(anvl::nv_reshape(k, c(b, nk, heads, head_dim)), perm)
        vh <- anvl::nv_transpose(anvl::nv_reshape(v, c(b, nk, heads, head_dim)), perm)
        attn <- yunque::yq_sdpa(qh, kh, vh, mask = mask, precision = precision)
        attn <- anvl::nv_reshape(anvl::nv_transpose(attn, perm), c(b, nq, inner))
        if (gated) {
            gates <- anvl::nv_logistic(gate_logits) * 2 # [B, nq, heads]
            a4 <- anvl::nv_reshape(attn, c(b, nq, heads, head_dim))
            g4 <- anvl::nv_broadcast_to(anvl::nv_unsqueeze(gates, 4L),
                                        c(b, nq, heads, head_dim))
            attn <- anvl::nv_reshape(a4 * g4, c(b, nq, inner))
        }
        yunque::yq_linear(attn, w$to_out, bias = w$to_out_b, precision = precision)
    }
}

#' LTX-2.3 timestep sinusoid (host-side)
#'
#' Parameter-free sinusoidal embedding matching
#' \code{diffuseR::ltx23_get_timestep_embedding} (cos-first,
#' \code{downscale_freq_shift = 0}, \code{max_period = 10000}). Computed
#' in base R (deterministic, no weights) and returned as an
#' \code{AnvlArray}, mirroring how the RoPE tables are precomputed outside
#' the model. Timestep values are NOT scaled here (the LTX pipeline scales
#' before calling the model).
#'
#' @param t Numeric vector of timestep values (row-major flatten of the
#'   per-token \code{[B, S]} timestep).
#' @param dim Integer. Sinusoid width (256).
#' @param max_period Numeric. Base period (10000).
#' @param device Character. Target device.
#'
#' @return AnvlArray \code{[length(t), dim]}, f32.
#'
#' @export
yq_ltx_time_sinusoid <- function(t, dim = 256L, max_period = 10000,
                                 device = "cpu") {
    t <- as.numeric(t)
    half <- dim %/% 2L
    exponent <- -log(max_period) * (0:(half - 1L)) / half
    freq <- exp(exponent)
    ang <- outer(t, freq) # [N, half]
    emb <- cbind(cos(ang), sin(ang)) # flip_sin_to_cos: cos then sin
    anvl::nv_array(emb, dtype = "f32", device = device)
}

# Round doubles to f32 precision (mirrors the reference $to(float32) casts
# on the rope frequencies/grid before cos/sin).
.yq_ltx_f32 <- function(x) {
    readBin(writeBin(as.double(x), raw(), size = 4L), "double",
            n = length(x), size = 4L)
}

# f32 rounding that preserves matrix/array dims.
.yq_ltx_f32m <- function(x) {
    r <- x
    r[] <- .yq_ltx_f32(x)
    r
}

# Build split-rope cos/sin AnvlArrays from a per-position frequency matrix
# freqs [N, dim/2]. The reference reshapes [B,N,dim/2] -> [B,N,H,r] ->
# [B,H,N,r]; here col (h-1)*r + j of row n lands at [b, h, n, j].
.yq_ltx_rope_tables <- function(freqs, num_heads, batch, device) {
    N <- nrow(freqs)
    r <- ncol(freqs) %/% num_heads
    to_arr <- function(mat) {
        a <- array(0, dim = c(batch, num_heads, N, r))
        for (h in seq_len(num_heads)) {
            for (j in seq_len(r)) {
                a[, h,, j] <- matrix(mat[, (h - 1L) * r + j], nrow = batch,
                                     ncol = N, byrow = TRUE)
            }
        }
        a
    }
    list(cos = anvl::nv_array(to_arr(cos(freqs)), dtype = "f32", device = device),
         sin = anvl::nv_array(to_arr(sin(freqs)), dtype = "f32", device = device))
}

# Per-position rope angles from normalized multi-axis coordinates
# grid [N, axes]: freqs[n, (s-1)*axes + a] = (2*grid[n,a]-1) * fr[s], with
# fr = theta^linspace(0,1,dim/(2*axes)) * pi/2. Matches the split branch of
# ltx23_rotary_pos_embed$forward (no padding when dim/2 is a multiple).
.yq_ltx_rope_freqs <- function(grid, dim, theta) {
    axes <- ncol(grid)
    steps <- as.integer(dim %/% (axes * 2L))
    fr <- .yq_ltx_f32(theta ^ seq(0, 1, length.out = steps) * pi / 2)
    g2 <- .yq_ltx_f32m(2 * grid - 1)
    freqs <- matrix(0, nrow(grid), steps * axes)
    for (s in seq_len(steps)) {
        for (a in seq_len(axes)) {
            freqs[, (s - 1L) * axes + a] <- .yq_ltx_f32(g2[, a] * fr[s])
        }
    }
    freqs
}

#' LTX-2.3 video RoPE cos/sin tables (host-side)
#'
#' Base-R port of \code{ltx23_rotary_pos_embed$prepare_video_coords} +
#' \code{$forward} for the "split" layout: patch-boundary coordinates
#' (frames scaled to seconds via fps with a causal temporal shift, height
#' and width in pixel space), midpoints normalized against the base grid,
#' outer product with the rope frequencies, reshaped per attention head.
#'
#' @param num_frames,height,width Integers. Latent token grid.
#' @param num_heads,head_dim Video attention shape (\code{dim = num_heads *
#'   head_dim}).
#' @param patch,patch_t Integers. Spatial/temporal patch sizes.
#' @param base_num_frames,base_height,base_width Integers. Normalization grid.
#' @param scale_factors Integer vector (time, height, width) VAE scales.
#' @param theta Numeric. RoPE theta.
#' @param causal_offset Integer. Causal-VAE temporal offset.
#' @param fps Numeric. Frames per second.
#' @param batch Integer. Batch size.
#' @param device Character. Target device.
#'
#' @return List \code{list(cos, sin)}, each \code{[batch, num_heads, N,
#'   head_dim/2]}, f32.
#'
#' @export
yq_ltx_video_rope <- function(num_frames, height, width, num_heads, head_dim,
                              patch = 1L, patch_t = 1L,
                              base_num_frames = 20L, base_height = 2048L,
                              base_width = 2048L,
                              scale_factors = c(8L, 32L, 32L), theta = 10000,
                              causal_offset = 1L, fps = 24, batch = 1L,
                              device = "cpu") {
    gf <- seq(0, num_frames - 1, by = patch_t)
    gh <- seq(0, height - 1, by = patch)
    gw <- seq(0, width - 1, by = patch)
    NF <- length(gf) ; NH <- length(gh) ; NW <- length(gw)
    f_c <- rep(gf, each = NH * NW)
    h_c <- rep(rep(gh, each = NW), times = NF)
    w_c <- rep(gw, times = NF * NH)
    starts <- cbind(f_c, h_c, w_c)
    ends <- cbind(f_c + patch_t, h_c + patch, w_c + patch)
    sc <- scale_factors
    starts <- sweep(starts, 2L, sc, `*`) ; ends <- sweep(ends, 2L, sc, `*`)
    tshift <- function(x) pmax(x + causal_offset - sc[1], 0) / fps
    starts[, 1] <- tshift(starts[, 1]) ; ends[, 1] <- tshift(ends[, 1])
    mid <- .yq_ltx_f32m((starts + ends) / 2)
    grid <- .yq_ltx_f32m(sweep(mid, 2L,
                               c(base_num_frames, base_height, base_width),
                               `/`))
    freqs <- .yq_ltx_rope_freqs(grid, num_heads * head_dim, theta)
    .yq_ltx_rope_tables(freqs, num_heads, batch, device)
}

#' LTX-2.3 audio RoPE cos/sin tables (host-side)
#'
#' Base-R port of \code{ltx23_rotary_pos_embed$prepare_audio_coords} +
#' \code{$forward} for the audio (1-D, seconds) modality.
#'
#' @param num_frames Integer. Audio latent token count.
#' @param num_heads,head_dim Audio attention shape.
#' @param patch_t Integer. Temporal patch size.
#' @param base_num_frames Integer. Normalization grid.
#' @param scale_factor Integer. Audio VAE scale factor.
#' @param sampling_rate,hop_length Integers. Mel spectrogram params.
#' @param theta Numeric. RoPE theta.
#' @param causal_offset Integer. Causal-VAE temporal offset.
#' @param batch Integer. Batch size.
#' @param device Character. Target device.
#'
#' @return List \code{list(cos, sin)}, each \code{[batch, num_heads, N,
#'   head_dim/2]}, f32.
#'
#' @export
yq_ltx_audio_rope <- function(num_frames, num_heads, head_dim, patch_t = 1L,
                              base_num_frames = 20L, scale_factor = 4L,
                              sampling_rate = 16000L, hop_length = 160L,
                              theta = 10000, causal_offset = 1L, batch = 1L,
                              device = "cpu") {
    gf <- seq(0, num_frames - 1, by = patch_t)
    to_s <- function(mel) pmax(mel + causal_offset - scale_factor, 0) *
    hop_length / sampling_rate
    start_s <- to_s(gf * scale_factor)
    end_s <- to_s((gf + patch_t) * scale_factor)
    mid <- .yq_ltx_f32((start_s + end_s) / 2)
    grid <- matrix(.yq_ltx_f32(mid / base_num_frames), ncol = 1L)
    freqs <- .yq_ltx_rope_freqs(grid, num_heads * head_dim, theta)
    .yq_ltx_rope_tables(freqs, num_heads, batch, device)
}

#' LTX-2.3 video->audio cross-attention RoPE tables (host-side)
#'
#' The audio<->video cross-attention rotates only the temporal axis. This
#' is the video side: a 1-axis rope over the video temporal midpoints
#' (normalized by \code{base_num_frames}) at the cross-attention dimension
#' and the video head count (\code{ltx23_transformer$cross_attn_rope}). The
#' audio-side cross rope is identical to \code{\link{yq_ltx_audio_rope}}
#' (same dim, heads, and 1-axis audio coords).
#'
#' @param num_frames,height,width Integers. Video latent token grid (only
#'   \code{num_frames} drives the temporal coordinate; height/width set N).
#' @param num_heads Integer. Video attention heads.
#' @param cross_dim Integer. Cross-attention dimension (audio inner dim).
#' @param patch,patch_t Integers. Spatial/temporal patch sizes.
#' @param base_num_frames Integer. Temporal normalization grid.
#' @param scale_factor Integer. VAE temporal scale (\code{scale_factors[1]}).
#' @param theta Numeric. RoPE theta.
#' @param causal_offset Integer. Causal-VAE temporal offset.
#' @param fps Numeric. Frames per second.
#' @param batch Integer. Batch size.
#' @param device Character. Target device.
#'
#' @return List \code{list(cos, sin)}, each \code{[batch, num_heads, N,
#'   cross_dim/(2*num_heads)]}, f32.
#'
#' @export
yq_ltx_video_cross_rope <- function(num_frames, height, width, num_heads,
                                    cross_dim, patch = 1L, patch_t = 1L,
                                    base_num_frames = 20L, scale_factor = 8L,
                                    theta = 10000, causal_offset = 1L,
                                    fps = 24, batch = 1L, device = "cpu") {
    NH <- length(seq(0, height - 1, by = patch))
    NW <- length(seq(0, width - 1, by = patch))
    f_c <- rep(seq(0, num_frames - 1, by = patch_t), each = NH * NW)
    tshift <- function(x) pmax(x + causal_offset - scale_factor, 0) / fps
    ts <- tshift(f_c * scale_factor)
    te <- tshift((f_c + patch_t) * scale_factor)
    mid <- .yq_ltx_f32((ts + te) / 2)
    grid <- matrix(.yq_ltx_f32(mid / base_num_frames), ncol = 1L)
    freqs <- .yq_ltx_rope_freqs(grid, cross_dim, theta)
    .yq_ltx_rope_tables(freqs, num_heads, batch, device)
}

#' LTX-2.3 text cross-attention mask (host-side)
#'
#' Converts a multiplicative padding mask \code{[B, S]} (1 real, 0 pad)
#' into the additive attention bias the score matrix wants, matching the
#' reference \code{(1 - mask) * -10000} expanded to \code{[B, 1, 1, S]} so
#' it broadcasts against scores \code{[B, H, Sq, S]} inside
#' \code{\link[yunque]{yq_sdpa}}.
#'
#' @param mask Numeric/integer matrix \code{[B, S]}, or NULL (no mask).
#' @param device Character. Target device.
#' @param neg Numeric. Masked-position bias (reference uses -10000).
#'
#' @return AnvlArray \code{[B, 1, 1, S]}, f32, or NULL.
#'
#' @export
yq_ltx_text_mask <- function(mask, device = "cpu", neg = -10000) {
    if (is.null(mask)) {
        return(NULL)
    }
    mask <- matrix(as.numeric(mask), nrow = nrow(mask))
    bias <- (1 - mask) * neg
    arr <- array(bias, dim = c(nrow(mask), 1L, 1L, ncol(mask)))
    anvl::nv_array(arr, dtype = "f32", device = device)
}

#' LTX-2.3 audio-video DiT forward (anvl port)
#'
#' Returns a closure over the static config; \code{anvl::jit()} it.
#' Activations, precomputed timestep sinusoids, and RoPE cos/sin tables
#' travel as three named lists plus the weights pytree.
#'
#' @param heads,head_dim Video attention shape (inner = heads*head_dim).
#' @param a_heads,a_head_dim Audio attention shape.
#' @param num_layers Integer. Transformer block count.
#' @param isolate Logical. \code{TRUE} runs the streams independently
#'   (\code{isolate_modalities}); \code{FALSE} adds a2v/v2a cross-attention.
#' @param eps Numeric. Norm epsilon.
#' @param precision Character. Matmul precision.
#'
#' @return Function of (act, sins, ropes, w):
#'   \itemize{
#'     \item act = list(hidden \code{[B,Sv,in]}, audio_hidden \code{[B,Sa,ain]},
#'       enc \code{[B,St,inner]}, audio_enc \code{[B,Sat,ainner]})
#'     \item sins = list(time \code{[B*Sv,256]}, audio_time \code{[B*Sa,256]},
#'       prompt \code{[B,256]}, audio_prompt \code{[B,256]})
#'     \item ropes = list(v_cos, v_sin, a_cos, a_sin, vca_cos, vca_sin,
#'       aca_cos, aca_sin)
#'     \item w weights pytree (\code{\link{yq_ltx_dit_load_weights}})
#'   }
#'   returning \code{list(video \code{[B,Sv,out]}, audio \code{[B,Sa,aout]})}.
#'
#' @export
yq_ltx_dit <- function(heads = 3L, head_dim = 8L, a_heads = 2L,
                       a_head_dim = 6L, num_layers = 2L, isolate = TRUE,
                       eps = 1e-6, precision = "highest") {
    inner <- heads * head_dim
    ainner <- a_heads * a_head_dim
    attn_self <- .yq_ltx_attn(heads, head_dim, TRUE, precision, eps)
    attn_aself <- .yq_ltx_attn(a_heads, a_head_dim, TRUE, precision, eps)
    attn_txt <- .yq_ltx_attn(heads, head_dim, TRUE, precision, eps)
    attn_atxt <- .yq_ltx_attn(a_heads, a_head_dim, TRUE, precision, eps)
    # a2v / v2a use the AUDIO attention shape (audio heads / head dim)
    attn_a2v <- .yq_ltx_attn(a_heads, a_head_dim, TRUE, precision, eps)
    attn_v2a <- .yq_ltx_attn(a_heads, a_head_dim, TRUE, precision, eps)

    ff <- function(x, w) {
        h1 <- yunque::yq_linear(x, w$net0, bias = w$net0_b,
                                precision = precision)
        yunque::yq_linear(.yq_ltx_gelu_tanh(h1), w$net2, bias = w$net2_b,
                          precision = precision)
    }
    rms <- function(x) yunque::yq_rms_norm(x, NULL, eps = eps)

    block <- function(h, ah, enc, aenc, temb, temb_a, temb_p, temb_ap,
                      vca_ss, vca_gate, aca_ss, aca_gate,
                      v_cos, v_sin, a_cos, a_sin, vca_cos, vca_sin, aca_cos, aca_sin,
                      enc_mask, aenc_mask, self_mask, wb) {
        sv <- anvl::shape(h)
        sa <- anvl::shape(ah)

        # 1.1 video self-attention
        va <- .yq_ltx_mod(temb, wb$sst, 9L, inner)
        nh <- rms(h) * (va[[2]] + 1) + va[[1]]
        h <- h + attn_self(nh, nh, wb$attn1, v_cos, v_sin, v_cos, v_sin,
                           self_mask) * va[[3]]

        # 1.2 audio self-attention
        aa <- .yq_ltx_mod(temb_a, wb$audio_sst, 9L, ainner)
        nah <- rms(ah) * (aa[[2]] + 1) + aa[[1]]
        ah <- ah + attn_aself(nah, nah, wb$audio_attn1, a_cos, a_sin, a_cos,
                              a_sin, NULL) * aa[[3]]

        # 2. prompt (text KV) modulation
        pa <- .yq_ltx_mod(temb_p, wb$prompt_sst, 2L, inner)
        apa <- .yq_ltx_mod(temb_ap, wb$audio_prompt_sst, 2L, ainner)

        # 2.1 video-text cross-attention
        nh <- rms(h) * (va[[8]] + 1) + va[[7]]
        enc_m <- enc * anvl::nv_broadcast_to(pa[[2]] + 1, anvl::shape(enc)) +
        anvl::nv_broadcast_to(pa[[1]], anvl::shape(enc))
        at <- attn_txt(nh, enc_m, wb$attn2, mask = enc_mask)
        h <- h + at * va[[9]]

        # 2.2 audio-text cross-attention
        nah <- rms(ah) * (aa[[8]] + 1) + aa[[7]]
        aenc_m <- aenc * anvl::nv_broadcast_to(apa[[2]] + 1, anvl::shape(aenc)) +
        anvl::nv_broadcast_to(apa[[1]], anvl::shape(aenc))
        aat <- attn_atxt(nah, aenc_m, wb$audio_attn2, mask = aenc_mask)
        ah <- ah + aat * aa[[9]]

        # 3. audio<->video cross-attention
        if (!isolate) {
            nh_ca <- rms(h) # audio_to_video_norm (weightless)
            nah_ca <- rms(ah) # video_to_audio_norm (weightless)
            vca <- .yq_ltx_mod(vca_ss, .yq_ltx_rows(wb$video_a2v, 1L, 4L), 4L, inner)
            aca <- .yq_ltx_mod(aca_ss, .yq_ltx_rows(wb$audio_a2v, 1L, 4L), 4L, ainner)
            a2v_gate <- .yq_ltx_mod(vca_gate, .yq_ltx_rows(wb$video_a2v, 5L, 5L),
                                    1L, inner)[[1]]
            v2a_gate <- .yq_ltx_mod(aca_gate, .yq_ltx_rows(wb$audio_a2v, 5L, 5L),
                                    1L, ainner)[[1]]

            # a2v: video query, audio key/value
            mvh <- nh_ca * (vca[[1]] + 1) + vca[[2]]
            mah <- nah_ca * (aca[[1]] + 1) + aca[[2]]
            a2v <- attn_a2v(mvh, mah, wb$a2v, vca_cos, vca_sin, aca_cos, aca_sin,
                            rope_heads_q = heads, rope_heads_k = a_heads)
            h <- h + a2v_gate * a2v

            # v2a: audio query, video key/value
            mvh2 <- nh_ca * (vca[[3]] + 1) + vca[[4]]
            mah2 <- nah_ca * (aca[[3]] + 1) + aca[[4]]
            v2a <- attn_v2a(mah2, mvh2, wb$v2a, aca_cos, aca_sin, vca_cos, vca_sin,
                            rope_heads_q = a_heads, rope_heads_k = heads)
            ah <- ah + v2a_gate * v2a
        }

        # 4. feed-forward
        h <- h + ff(rms(h) * (va[[5]] + 1) + va[[4]], wb$ff) * va[[6]]
        ah <- ah + ff(rms(ah) * (aa[[5]] + 1) + aa[[4]], wb$audio_ff) * aa[[6]]

        list(h, ah)
    }

    function(act, sins, ropes, w) {
        hidden <- act$hidden; audio_hidden <- act$audio_hidden
        enc <- act$enc; aenc <- act$audio_enc
        b <- anvl::shape(hidden)[1L]
        sv <- anvl::shape(hidden)[2L]
        sa <- anvl::shape(audio_hidden)[2L]

        # 2. input projections
        h <- yunque::yq_linear(hidden, w$proj_in, bias = w$proj_in_b,
                               precision = precision)
        ah <- yunque::yq_linear(audio_hidden, w$audio_proj_in,
                                bias = w$audio_proj_in_b, precision = precision)

        # 3. timestep embeddings + global modulation
        te <- .yq_ltx_adaln(sins$time, w$time_embed, precision)
        ate <- .yq_ltx_adaln(sins$audio_time, w$audio_time_embed, precision)
        temb <- anvl::nv_reshape(te$mod, c(b, sv, 9L * inner))
        temb_a <- anvl::nv_reshape(ate$mod, c(b, sa, 9L * ainner))
        emb_v <- anvl::nv_reshape(te$emb, c(b, sv, inner))
        emb_a <- anvl::nv_reshape(ate$emb, c(b, sa, ainner))

        pe <- .yq_ltx_adaln(sins$prompt, w$prompt_adaln, precision)
        ape <- .yq_ltx_adaln(sins$audio_prompt, w$audio_prompt_adaln, precision)
        temb_p <- anvl::nv_reshape(pe$mod, c(b, 1L, 2L * inner))
        temb_ap <- anvl::nv_reshape(ape$mod, c(b, 1L, 2L * ainner))

        vca_ss <- vca_gate <- aca_ss <- aca_gate <- NULL
        if (!isolate) {
            # gate_scale_factor = 1 (multipliers equal), use_cross_timestep
            # FALSE: av modulation reuses the per-token time sinusoids.
            vss <- .yq_ltx_adaln(sins$time, w$av_video_ss, precision)
            vg <- .yq_ltx_adaln(sins$time, w$av_video_gate, precision)
            ass <- .yq_ltx_adaln(sins$audio_time, w$av_audio_ss, precision)
            ag <- .yq_ltx_adaln(sins$audio_time, w$av_audio_gate, precision)
            vca_ss <- anvl::nv_reshape(vss$mod, c(b, sv, 4L * inner))
            vca_gate <- anvl::nv_reshape(vg$mod, c(b, sv, 1L * inner))
            aca_ss <- anvl::nv_reshape(ass$mod, c(b, sa, 4L * ainner))
            aca_gate <- anvl::nv_reshape(ag$mod, c(b, sa, 1L * ainner))
        }

        # 5. transformer blocks
        for (i in seq_len(num_layers)) {
            res <- block(h, ah, enc, aenc, temb, temb_a, temb_p, temb_ap,
                         vca_ss, vca_gate, aca_ss, aca_gate,
                         ropes$v_cos, ropes$v_sin, ropes$a_cos, ropes$a_sin,
                         ropes$vca_cos, ropes$vca_sin, ropes$aca_cos, ropes$aca_sin,
                         act$enc_mask, act$audio_enc_mask, act$self_mask, w$blocks[[i]])
            h <- res[[1]]; ah <- res[[2]]
        }

        # 6. output layers (per-token scale/shift from embedded timestep)
        out_mod <- function(x, table, embt, sdim, S) {
            shift <- anvl::nv_broadcast_to(anvl::nv_reshape(.yq_ltx_row(table, 1L),
                    c(1L, 1L, sdim)),
                c(b, S, sdim)) + embt
            scale <- anvl::nv_broadcast_to(anvl::nv_reshape(.yq_ltx_row(table, 2L),
                    c(1L, 1L, sdim)),
                c(b, S, sdim)) + embt
            yunque::yq_layer_norm(x, eps = 1e-6) * (scale + 1) + shift
        }
        vo <- out_mod(h, w$scale_shift_table, emb_v, inner, sv)
        ao <- out_mod(ah, w$audio_scale_shift_table, emb_a, ainner, sa)
        list(video = yunque::yq_linear(vo, w$proj_out, bias = w$proj_out_b,
                                       precision = precision),
             audio = yunque::yq_linear(ao, w$audio_proj_out, bias = w$audio_proj_out_b,
                                       precision = precision))
    }
}

#' Load LTX-2.3 DiT weights into an anvl pytree
#'
#' Reads a \code{ltx23_transformer} state-dict safetensors file (f32),
#' transposing 2-D linears to \code{[in, out]} and wrapping each tensor as
#' an \code{AnvlArray} on \code{device}. Modulation tables (scale/shift)
#' load in their raw \code{[num_params, dim]} shape. Block count is derived
#' from the \code{transformer_blocks.N} key census.
#'
#' @param path Path to the state-dict safetensors file.
#' @param isolate Logical. If \code{FALSE}, also load the a2v/v2a
#'   cross-attention and global av-modulation weights.
#' @param device Character. Target device.
#'
#' @return Weights pytree for \code{\link{yq_ltx_dit}}.
#'
#' @export
yq_ltx_dit_load_weights <- function(path, isolate = TRUE, device = "cpu") {
    st <- yunque::yq_st_open(path)
    on.exit(close(st$con))
    lin <- function(key) anvl::nv_array(yunque::yq_st_read(st, key,
            transpose = TRUE),
                                        dtype = "f32", device = device)
    vec <- function(key) anvl::nv_array(yunque::yq_st_read(st, key),
                                        dtype = "f32", device = device)
    tab <- function(key) anvl::nv_array(yunque::yq_st_read(st, key, transpose = FALSE),
                                        dtype = "f32", device = device)

    adaln <- function(p) list(
                              l1 = lin(paste0(p, ".emb.timestep_embedder.linear_1.weight")),
                              l1b = vec(paste0(p, ".emb.timestep_embedder.linear_1.bias")),
                              l2 = lin(paste0(p, ".emb.timestep_embedder.linear_2.weight")),
                              l2b = vec(paste0(p, ".emb.timestep_embedder.linear_2.bias")),
                              lin = lin(paste0(p, ".linear.weight")),
                              linb = vec(paste0(p, ".linear.bias"))
    )
    attn <- function(p, gated = TRUE) {
        a <- list(
                  norm_q = vec(paste0(p, ".norm_q.weight")),
                  norm_k = vec(paste0(p, ".norm_k.weight")),
                  to_q = lin(paste0(p, ".to_q.weight")), to_q_b = vec(paste0(p, ".to_q.bias")),
                  to_k = lin(paste0(p, ".to_k.weight")), to_k_b = vec(paste0(p, ".to_k.bias")),
                  to_v = lin(paste0(p, ".to_v.weight")), to_v_b = vec(paste0(p, ".to_v.bias")),
                  to_out = lin(paste0(p, ".to_out.0.weight")),
                  to_out_b = vec(paste0(p, ".to_out.0.bias"))
        )
        if (gated) {
            a$gate <- lin(paste0(p, ".to_gate_logits.weight"))
            a$gate_b <- vec(paste0(p, ".to_gate_logits.bias"))
        }
        a
    }
    ff <- function(p) list(
                           net0 = lin(paste0(p, ".net.0.proj.weight")),
                           net0_b = vec(paste0(p, ".net.0.proj.bias")),
                           net2 = lin(paste0(p, ".net.2.weight")),
                           net2_b = vec(paste0(p, ".net.2.bias"))
    )

    n_layers <- {
        m <- regmatches(names(st$header),
                        regexpr("^transformer_blocks\\.[0-9]+\\.", names(st$header)))
        idx <- as.integer(sub("^transformer_blocks\\.([0-9]+)\\.$", "\\1", m))
        if (length(idx) == 0L) {
            0L
        } else {
            max(idx) + 1L
        }
    }

    block <- function(i) {
        p <- sprintf("transformer_blocks.%d", i)
        wb <- list(
                   sst = tab(paste0(p, ".scale_shift_table")),
                   audio_sst = tab(paste0(p, ".audio_scale_shift_table")),
                   prompt_sst = tab(paste0(p, ".prompt_scale_shift_table")),
                   audio_prompt_sst = tab(paste0(p, ".audio_prompt_scale_shift_table")),
                   attn1 = attn(paste0(p, ".attn1")),
                   audio_attn1 = attn(paste0(p, ".audio_attn1")),
                   attn2 = attn(paste0(p, ".attn2")),
                   audio_attn2 = attn(paste0(p, ".audio_attn2")),
                   ff = ff(paste0(p, ".ff")),
                   audio_ff = ff(paste0(p, ".audio_ff"))
        )
        if (!isolate) {
            wb$video_a2v <- tab(paste0(p, ".video_a2v_cross_attn_scale_shift_table"))
            wb$audio_a2v <- tab(paste0(p, ".audio_a2v_cross_attn_scale_shift_table"))
            wb$a2v <- attn(paste0(p, ".audio_to_video_attn"))
            wb$v2a <- attn(paste0(p, ".video_to_audio_attn"))
        }
        wb
    }

    w <- list(
              proj_in = lin("proj_in.weight"), proj_in_b = vec("proj_in.bias"),
              audio_proj_in = lin("audio_proj_in.weight"),
              audio_proj_in_b = vec("audio_proj_in.bias"),
              time_embed = adaln("time_embed"),
              audio_time_embed = adaln("audio_time_embed"),
              prompt_adaln = adaln("prompt_adaln"),
              audio_prompt_adaln = adaln("audio_prompt_adaln"),
              scale_shift_table = tab("scale_shift_table"),
              audio_scale_shift_table = tab("audio_scale_shift_table"),
              proj_out = lin("proj_out.weight"), proj_out_b = vec("proj_out.bias"),
              audio_proj_out = lin("audio_proj_out.weight"),
              audio_proj_out_b = vec("audio_proj_out.bias"),
              blocks = lapply(seq_len(n_layers) - 1L, block)
    )
    if (!isolate) {
        w$av_video_ss <- adaln("av_cross_attn_video_scale_shift")
        w$av_video_gate <- adaln("av_cross_attn_video_a2v_gate")
        w$av_audio_ss <- adaln("av_cross_attn_audio_scale_shift")
        w$av_audio_gate <- adaln("av_cross_attn_audio_v2a_gate")
    }
    w
}
