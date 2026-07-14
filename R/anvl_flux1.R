#' FLUX.1 MMDiT transformer (anvl port)
#'
#' anvl re-implementation of \code{diffuseR::flux_transformer}
#' (FluxTransformer2DModel, FLUX.1-schnell). Same double-stream MMDiT +
#' single-stream family as the FLUX.2 port
#' (\code{\link{yq_flux2_transformer}}) and reuses its helpers, but with
#' three deliberate differences from FLUX.2:
#' \enumerate{
#'   \item \strong{Per-block modulation (adaLN-Zero).} FLUX.2 computes
#'     the (shift, scale, gate) modulation once and shares it across
#'     every block; FLUX.1 computes it inside each block from
#'     \code{silu(temb)} via that block's own \code{norm1}/
#'     \code{norm1_context}/\code{norm} linear.
#'   \item \strong{No guidance embedding (schnell is CFG-free).} The
#'     timestep conditioning is \code{timestep_embedder(sinusoid) +
#'     text_embedder(pooled CLIP)} with no guidance term.
#'   \item \strong{Non-fused single-stream block, GELU-tanh FFN.} FLUX.1
#'     keeps separate \code{to_q/to_k/to_v} + \code{proj_mlp} and a
#'     \code{proj_out} over \code{cat(attn, mlp)} (FLUX.2 fuses
#'     QKV+MLP), and its feed-forward is GELU-tanh (FLUX.2 is SwiGLU).
#'     Every linear carries a bias (FLUX.2's are bias-free).
#' }
#' Weights travel as a nested named list (pytree); see
#' \code{\link{yq_flux1_load_weights}}. The timestep sinusoid
#' (\code{\link{yq_flux1_time_embed}}) and RoPE cos/sin tables are
#' precomputed host-side and passed as inputs, matching the diffusers
#' pipeline boundary and the FLUX.2 port.
#'
#' @name anvl_flux1
NULL

#' FLUX.1 timestep sinusoidal projection (host-side)
#'
#' Parameter-free sinusoidal embedding of \code{timestep * 1000},
#' matching \code{diffuseR::ltx23_get_timestep_embedding} with
#' \code{flip_sin_to_cos = TRUE}, \code{downscale_freq_shift = 0}
#' (identical to \code{\link{yq_flux2_time_proj}}). Computed in base R and
#' returned as an \code{AnvlArray}.
#'
#' @param timestep Numeric vector (sigma space, 0-1); scaled by 1000
#'   internally.
#' @param dim Integer. Projection width (256).
#' @param max_period Numeric. Base period (10000).
#' @param device Character. Target device.
#'
#' @return AnvlArray \code{[length(timestep), dim]}, f32.
#'
#' @export
yq_flux1_time_embed <- function(timestep, dim = 256L, max_period = 10000,
                                device = "cpu") {
    t <- as.numeric(timestep) * 1000
    half <- dim %/% 2L
    exponent <- -log(max_period) * (0:(half - 1L)) / half
    freq <- exp(exponent)
    ang <- outer(t, freq)                     # [N, half]
    emb <- cbind(cos(ang), sin(ang))          # flip_sin_to_cos: cos then sin
    anvl::nv_array(emb, dtype = "f32", device = device)
}

# GELU with the tanh approximation (torch nnf_gelu(approximate = "tanh")):
# 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))). Pure f32 in
# anvl, so the R-scalar constants promote cleanly (no f16 concern).
.yq_flux1_gelu_tanh <- function(x) {
    c0 <- sqrt(2 / pi)
    inner <- (x + (x * x * x) * 0.044715) * c0
    x * (anvl::nv_tanh(inner) + 1) * 0.5
}

# Bias-aware per-head RMS-normed projection: linear(+bias) -> [B,S,H,D]
# -> optional rms(head). FLUX.1 linears all carry a bias (unlike the
# bias-free FLUX.2 .yq_qkv_head), hence a distinct name (one name, one
# signature, package-wide).
.yq_flux1_qkv_head <- function(x, w_t, bias, norm_w, heads, head_dim, eps,
                               precision) {
    s <- anvl::shape(x)
    proj <- yunque::linear(x, w_t, bias = bias, precision = precision)
    r <- anvl::nv_reshape(proj, c(s[1L], s[2L], heads, head_dim))
    if (!is.null(norm_w)) {
        r <- yunque::rms_norm(r, norm_w, eps = eps)
    }
    r
}

#' FLUX.1 double-stream (MMDiT) block (anvl port)
#'
#' anvl re-implementation of \code{diffuseR::flux_double_block} +
#' \code{flux_attention(added_kv = TRUE)}: separate image and text
#' streams, each with per-block adaLN-Zero modulation (computed inside
#' the block from \code{silu(temb)}) and a GELU-tanh feed-forward, joined
#' by a single attention over the concatenated [text; image] sequence
#' (text tokens first, matching the rotary layout).
#'
#' Returns a closure over the static config; \code{anvl::jit()} the
#' enclosing transformer. Weights travel as a named list mirroring the
#' checkpoint keys under \code{transformer_blocks.N.}.
#'
#' @param heads Integer. Attention heads.
#' @param head_dim Integer. Per-head dimension.
#' @param eps Numeric. Norm epsilon.
#' @param precision Character. Matmul precision (see
#'   \code{\link[yunque]{yq_linear}}).
#'
#' @return Function of (h, c, sil, cos, sin, w) returning
#'   \code{list(c_out, h_out)}.
#'
#' @export
yq_flux1_double_block <- function(heads = 24L, head_dim = 128L, eps = 1e-6,
                                  precision = "highest") {
    heads <- as.integer(heads)
    head_dim <- as.integer(head_dim)
    inner <- heads * head_dim

    ff <- function(x, w_in, b_in, w_out, b_out) {
        h <- yunque::linear(x, w_in, bias = b_in, precision = precision)
        h <- .yq_flux1_gelu_tanh(h)
        yunque::linear(h, w_out, bias = b_out, precision = precision)
    }

    function(h, c, sil, cos, sin, w) {
        dim <- anvl::shape(h)[3L]
        s_txt <- anvl::shape(c)[2L]
        s_img <- anvl::shape(h)[2L]
        b <- anvl::shape(h)[1L]

        # Per-block modulation (adaLN-Zero): [N, 6*dim] each stream.
        mod_img <- yunque::linear(sil, w$norm1_lin, bias = w$norm1_lin_b,
                                     precision = precision)
        mod_txt <- yunque::linear(sil, w$norm1c_lin, bias = w$norm1c_lin_b,
                                     precision = precision)
        mi <- .yq_mod_split(mod_img, 2L, dim)
        mt <- .yq_mod_split(mod_txt, 2L, dim)
        msa <- mi[[1L]]; mmlp <- mi[[2L]]
        cmsa <- mt[[1L]]; cmlp <- mt[[2L]]

        hs <- anvl::shape(h)
        cs <- anvl::shape(c)
        modulate <- function(x, shift, scale) {
            sh <- anvl::shape(x)
            yunque::layer_norm(x, eps = eps) *
            anvl::nv_broadcast_to(scale + 1, sh) +
            anvl::nv_broadcast_to(shift, sh)
        }

        norm_h <- modulate(h, msa[[1L]], msa[[2L]])
        norm_c <- modulate(c, cmsa[[1L]], cmsa[[2L]])

        # Joint attention. Per-head [B, S, H, D], text q/k/v first.
        q_i <- .yq_flux1_qkv_head(norm_h, w$to_q, w$to_q_b, w$norm_q,
                                  heads, head_dim, eps, precision)
        k_i <- .yq_flux1_qkv_head(norm_h, w$to_k, w$to_k_b, w$norm_k,
                                  heads, head_dim, eps, precision)
        v_i <- .yq_flux1_qkv_head(norm_h, w$to_v, w$to_v_b, NULL,
                                  heads, head_dim, eps, precision)
        q_t <- .yq_flux1_qkv_head(norm_c, w$add_q_proj, w$add_q_proj_b,
                                  w$norm_added_q, heads, head_dim, eps, precision)
        k_t <- .yq_flux1_qkv_head(norm_c, w$add_k_proj, w$add_k_proj_b,
                                  w$norm_added_k, heads, head_dim, eps, precision)
        v_t <- .yq_flux1_qkv_head(norm_c, w$add_v_proj, w$add_v_proj_b, NULL,
                                  heads, head_dim, eps, precision)

        q <- anvl::nv_concatenate(q_t, q_i, dimension = 2L)
        k <- anvl::nv_concatenate(k_t, k_i, dimension = 2L)
        v <- anvl::nv_concatenate(v_t, v_i, dimension = 2L)

        perm <- c(1L, 3L, 2L, 4L)             # [B, S, H, D] -> [B, H, S, D]
        q <- anvl::nv_transpose(q, perm)
        k <- anvl::nv_transpose(k, perm)
        v <- anvl::nv_transpose(v, perm)

        s_all <- s_txt + s_img
        chs <- c(b, heads, s_all, head_dim)
        cb <- anvl::nv_broadcast_to(cos, chs)
        sb <- anvl::nv_broadcast_to(sin, chs)
        q <- yunque::rope_apply(q, cb, sb)
        k <- yunque::rope_apply(k, cb, sb)

        attn <- yunque::sdpa(q, k, v, precision = precision)
        attn <- anvl::nv_reshape(anvl::nv_transpose(attn, perm),
                                 c(b, s_all, inner))

        ctx <- yunque::slice_seq(attn, 1L, s_txt)
        img <- yunque::slice_seq(attn, s_txt + 1L, s_all)
        attn_img <- yunque::linear(img, w$to_out, bias = w$to_out_b,
                                      precision = precision)
        attn_ctx <- yunque::linear(ctx, w$to_add_out, bias = w$to_add_out_b,
                                      precision = precision)

        h <- h + anvl::nv_broadcast_to(msa[[3L]], hs) * attn_img
        norm_h2 <- modulate(h, mmlp[[1L]], mmlp[[2L]])
        h <- h + anvl::nv_broadcast_to(mmlp[[3L]], hs) *
        ff(norm_h2, w$ff_in, w$ff_in_b, w$ff_out, w$ff_out_b)

        c <- c + anvl::nv_broadcast_to(cmsa[[3L]], cs) * attn_ctx
        norm_c2 <- modulate(c, cmlp[[1L]], cmlp[[2L]])
        c <- c + anvl::nv_broadcast_to(cmlp[[3L]], cs) *
        ff(norm_c2, w$ff_c_in, w$ff_c_in_b, w$ff_c_out, w$ff_c_out_b)

        list(c, h)
    }
}

#' FLUX.1 single-stream block (anvl port)
#'
#' anvl re-implementation of \code{diffuseR::flux_single_block} +
#' \code{flux_attention(pre_only = TRUE)}: per-block adaLN-Zero
#' modulation (shift, scale, gate) over a parameterless LayerNorm,
#' \strong{non-fused} attention (\code{to_q/to_k/to_v}) and a parallel
#' GELU-tanh MLP (\code{proj_mlp}), joined by a single \code{proj_out}
#' over \code{cat(attn, mlp)} with a shared output gate.
#'
#' @param heads Integer. Attention heads.
#' @param head_dim Integer. Per-head dimension.
#' @param mlp_ratio Numeric. MLP hidden multiplier (4.0).
#' @param eps Numeric. Norm epsilon.
#' @param precision Character. Matmul precision.
#'
#' @return Function of (h, sil, cos, sin, w) returning the joint hidden
#'   states \code{[B, S, dim]}.
#'
#' @export
yq_flux1_single_block <- function(heads = 24L, head_dim = 128L,
                                  mlp_ratio = 4.0, eps = 1e-6,
                                  precision = "highest") {
    heads <- as.integer(heads)
    head_dim <- as.integer(head_dim)
    inner <- heads * head_dim

    function(h, sil, cos, sin, w) {
        s <- anvl::shape(h)
        b <- s[1L]
        n <- s[2L]
        dim <- s[3L]

        mod <- yunque::linear(sil, w$norm_lin, bias = w$norm_lin_b,
                                 precision = precision)
        ms <- .yq_mod_split(mod, 1L, dim)[[1L]]  # shift, scale, gate

        norm_h <- yunque::layer_norm(h, eps = eps) *
        anvl::nv_broadcast_to(ms[[2L]] + 1, s) +
        anvl::nv_broadcast_to(ms[[1L]], s)

        mlp <- .yq_flux1_gelu_tanh(
            yunque::linear(norm_h, w$proj_mlp, bias = w$proj_mlp_b,
                              precision = precision))

        q <- .yq_flux1_qkv_head(norm_h, w$to_q, w$to_q_b, w$norm_q,
                                heads, head_dim, eps, precision)
        k <- .yq_flux1_qkv_head(norm_h, w$to_k, w$to_k_b, w$norm_k,
                                heads, head_dim, eps, precision)
        v <- .yq_flux1_qkv_head(norm_h, w$to_v, w$to_v_b, NULL,
                                heads, head_dim, eps, precision)
        perm <- c(1L, 3L, 2L, 4L)
        q <- anvl::nv_transpose(q, perm)
        k <- anvl::nv_transpose(k, perm)
        v <- anvl::nv_transpose(v, perm)

        chs <- c(b, heads, n, head_dim)
        cb <- anvl::nv_broadcast_to(cos, chs)
        sb <- anvl::nv_broadcast_to(sin, chs)
        q <- yunque::rope_apply(q, cb, sb)
        k <- yunque::rope_apply(k, cb, sb)

        attn <- yunque::sdpa(q, k, v, precision = precision)
        attn <- anvl::nv_reshape(anvl::nv_transpose(attn, perm),
                                 c(b, n, inner))

        out <- yunque::linear(
            anvl::nv_concatenate(attn, mlp, dimension = 3L),
            w$proj_out, bias = w$proj_out_b, precision = precision)
        h + anvl::nv_broadcast_to(ms[[3L]], s) * out
    }
}

#' FLUX.1 transformer forward (anvl port)
#'
#' anvl re-implementation of \code{diffuseR::flux_transformer} forward:
#' x/context embedders, timestep + pooled-text conditioning, a stack of
#' double-stream blocks over separate text/image streams, then
#' single-stream blocks over the concatenated sequence, an
#' adaLN-continuous output norm, and the velocity projection.
#'
#' Returns a closure over the static config; \code{anvl::jit()} it.
#' Weights travel as a named pytree (see \code{\link{yq_flux1_load_weights}}).
#' The timestep sinusoid (\code{\link{yq_flux1_time_embed}}) and RoPE
#' tables are precomputed host-side and passed as inputs.
#'
#' @param num_layers Integer. Double-stream blocks.
#' @param num_single_layers Integer. Single-stream blocks.
#' @param heads Integer. Attention heads.
#' @param head_dim Integer. Per-head dim.
#' @param mlp_ratio Numeric. Single-block MLP multiplier (4.0).
#' @param eps Numeric. Norm epsilon.
#' @param precision Character. Matmul precision.
#'
#' @return Function of (latents, text_embeds, pooled, time_sin, cos, sin,
#'   w):
#'   \itemize{
#'     \item latents \code{[B, S_img, in_channels]}
#'     \item text_embeds \code{[B, S_txt, joint_dim]}
#'     \item pooled \code{[B, pooled_dim]} pooled CLIP projection
#'     \item time_sin \code{[B, 256]} from \code{yq_flux1_time_embed()}
#'     \item cos, sin \code{[S_txt + S_img, head_dim]} RoPE tables
#'     \item w weights pytree
#'   }
#'   returning velocity \code{[B, S_img, out_channels]}.
#'
#' @export
yq_flux1_transformer <- function(num_layers = 19L, num_single_layers = 38L,
                                 heads = 24L, head_dim = 128L,
                                 mlp_ratio = 4.0, eps = 1e-6,
                                 precision = "highest") {
    double_blk <- yq_flux1_double_block(heads, head_dim, eps, precision)
    single_blk <- yq_flux1_single_block(heads, head_dim, mlp_ratio, eps,
                                        precision)

    function(latents, text_embeds, pooled, time_sin, cos, sin, w) {
        x <- yunque::linear(latents, w$x_embedder, bias = w$x_embedder_b,
                               precision = precision)
        cc <- yunque::linear(text_embeds, w$context_embedder,
                                bias = w$context_embedder_b, precision = precision)
        dim <- anvl::shape(x)[3L]

        # temb = timestep_embedder(sinusoid) + text_embedder(pooled)
        ts <- yunque::linear(
            yunque::silu(yunque::linear(time_sin, w$ts_1, bias = w$ts_1_b,
                                              precision = precision)),
            w$ts_2, bias = w$ts_2_b, precision = precision)
        txt <- yunque::linear(
            yunque::silu(yunque::linear(pooled, w$txt_1, bias = w$txt_1_b,
                                              precision = precision)),
            w$txt_2, bias = w$txt_2_b, precision = precision)
        temb <- ts + txt
        sil <- yunque::silu(temb)

        for (i in seq_len(num_layers)) {
            res <- double_blk(x, cc, sil, cos, sin, w$double[[i]])
            cc <- res[[1L]]
            x <- res[[2L]]
        }

        s_txt <- anvl::shape(cc)[2L]
        hs <- anvl::nv_concatenate(cc, x, dimension = 2L)
        for (i in seq_len(num_single_layers)) {
            hs <- single_blk(hs, sil, cos, sin, w$single[[i]])
        }
        s_all <- anvl::shape(hs)[2L]
        hs <- yunque::slice_seq(hs, s_txt + 1L, s_all)

        # adaLN-continuous output norm: scale first, then shift.
        no <- yunque::linear(sil, w$norm_out, bias = w$norm_out_b,
                                precision = precision)
        scale <- yunque::slice_lastdim(no, 1L, dim)
        shift <- yunque::slice_lastdim(no, dim + 1L, 2L * dim)
        sh <- anvl::shape(hs)
        s2 <- anvl::shape(scale)
        scale <- anvl::nv_reshape(scale, c(s2[1L], 1L, s2[2L]))
        shift <- anvl::nv_reshape(shift, c(s2[1L], 1L, s2[2L]))
        hs <- yunque::layer_norm(hs, eps = eps) *
        anvl::nv_broadcast_to(scale + 1, sh) +
        anvl::nv_broadcast_to(shift, sh)

        yunque::linear(hs, w$proj_out, bias = w$proj_out_b,
                          precision = precision)
    }
}

#' Load FLUX.1 transformer weights into an anvl pytree
#'
#' Reads every FLUX.1 transformer weight from a safetensors file (f32),
#' transposing 2-D linears to \code{[in, out]} and keeping biases / norm
#' weights as vectors, and wraps each as an \code{AnvlArray} on
#' \code{device}. The file is the random-init state-dict fixture written
#' by \code{tools/gen_fixture_flux1_dit.R} (which also carries the
#' inputs/output under \code{input.*}/\code{output}, ignored here);
#' the same loader works on a real diffusers checkpoint (identical key
#' tree). Returns the nested list \code{\link{yq_flux1_transformer}}
#' expects. Strict census: every named key must exist.
#'
#' @param path Path to the safetensors file holding the state dict.
#' @param num_layers Integer. Double-stream blocks.
#' @param num_single_layers Integer. Single-stream blocks.
#' @param device Character. Target device.
#'
#' @return Weights pytree.
#'
#' @export
yq_flux1_load_weights <- function(path, num_layers = 19L,
                                  num_single_layers = 38L, device = "cpu") {
    st <- yunque::st_open(path)
    on.exit(close(st$con))
    lin <- function(key) anvl::nv_array(
        yunque::st_read(st, key, transpose = TRUE),
        dtype = "f32", device = device)
    vec <- function(key) anvl::nv_array(
        yunque::st_read(st, key), dtype = "f32", device = device)

    tt <- "time_text_embed."
    w <- list(
        x_embedder = lin("x_embedder.weight"),
        x_embedder_b = vec("x_embedder.bias"),
        context_embedder = lin("context_embedder.weight"),
        context_embedder_b = vec("context_embedder.bias"),
        ts_1 = lin(paste0(tt, "timestep_embedder.linear_1.weight")),
        ts_1_b = vec(paste0(tt, "timestep_embedder.linear_1.bias")),
        ts_2 = lin(paste0(tt, "timestep_embedder.linear_2.weight")),
        ts_2_b = vec(paste0(tt, "timestep_embedder.linear_2.bias")),
        txt_1 = lin(paste0(tt, "text_embedder.linear_1.weight")),
        txt_1_b = vec(paste0(tt, "text_embedder.linear_1.bias")),
        txt_2 = lin(paste0(tt, "text_embedder.linear_2.weight")),
        txt_2_b = vec(paste0(tt, "text_embedder.linear_2.bias")),
        norm_out = lin("norm_out.linear.weight"),
        norm_out_b = vec("norm_out.linear.bias"),
        proj_out = lin("proj_out.weight"),
        proj_out_b = vec("proj_out.bias")
    )

    w$double <- lapply(seq_len(num_layers) - 1L, function(i) {
        p <- sprintf("transformer_blocks.%d.", i)
        a <- paste0(p, "attn.")
        list(
            norm1_lin = lin(paste0(p, "norm1.linear.weight")),
            norm1_lin_b = vec(paste0(p, "norm1.linear.bias")),
            norm1c_lin = lin(paste0(p, "norm1_context.linear.weight")),
            norm1c_lin_b = vec(paste0(p, "norm1_context.linear.bias")),
            to_q = lin(paste0(a, "to_q.weight")),
            to_q_b = vec(paste0(a, "to_q.bias")),
            to_k = lin(paste0(a, "to_k.weight")),
            to_k_b = vec(paste0(a, "to_k.bias")),
            to_v = lin(paste0(a, "to_v.weight")),
            to_v_b = vec(paste0(a, "to_v.bias")),
            norm_q = vec(paste0(a, "norm_q.weight")),
            norm_k = vec(paste0(a, "norm_k.weight")),
            add_q_proj = lin(paste0(a, "add_q_proj.weight")),
            add_q_proj_b = vec(paste0(a, "add_q_proj.bias")),
            add_k_proj = lin(paste0(a, "add_k_proj.weight")),
            add_k_proj_b = vec(paste0(a, "add_k_proj.bias")),
            add_v_proj = lin(paste0(a, "add_v_proj.weight")),
            add_v_proj_b = vec(paste0(a, "add_v_proj.bias")),
            norm_added_q = vec(paste0(a, "norm_added_q.weight")),
            norm_added_k = vec(paste0(a, "norm_added_k.weight")),
            to_out = lin(paste0(a, "to_out.0.weight")),
            to_out_b = vec(paste0(a, "to_out.0.bias")),
            to_add_out = lin(paste0(a, "to_add_out.weight")),
            to_add_out_b = vec(paste0(a, "to_add_out.bias")),
            ff_in = lin(paste0(p, "ff.net.0.proj.weight")),
            ff_in_b = vec(paste0(p, "ff.net.0.proj.bias")),
            ff_out = lin(paste0(p, "ff.net.2.weight")),
            ff_out_b = vec(paste0(p, "ff.net.2.bias")),
            ff_c_in = lin(paste0(p, "ff_context.net.0.proj.weight")),
            ff_c_in_b = vec(paste0(p, "ff_context.net.0.proj.bias")),
            ff_c_out = lin(paste0(p, "ff_context.net.2.weight")),
            ff_c_out_b = vec(paste0(p, "ff_context.net.2.bias"))
        )
    })

    w$single <- lapply(seq_len(num_single_layers) - 1L, function(i) {
        p <- sprintf("single_transformer_blocks.%d.", i)
        a <- paste0(p, "attn.")
        list(
            norm_lin = lin(paste0(p, "norm.linear.weight")),
            norm_lin_b = vec(paste0(p, "norm.linear.bias")),
            proj_mlp = lin(paste0(p, "proj_mlp.weight")),
            proj_mlp_b = vec(paste0(p, "proj_mlp.bias")),
            proj_out = lin(paste0(p, "proj_out.weight")),
            proj_out_b = vec(paste0(p, "proj_out.bias")),
            to_q = lin(paste0(a, "to_q.weight")),
            to_q_b = vec(paste0(a, "to_q.bias")),
            to_k = lin(paste0(a, "to_k.weight")),
            to_k_b = vec(paste0(a, "to_k.bias")),
            to_v = lin(paste0(a, "to_v.weight")),
            to_v_b = vec(paste0(a, "to_v.bias")),
            norm_q = vec(paste0(a, "norm_q.weight")),
            norm_k = vec(paste0(a, "norm_k.weight"))
        )
    })

    w
}
