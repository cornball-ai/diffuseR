#' Split a shared modulation tensor into (shift, scale, gate) triples
#'
#' Mirrors diffusers \code{Flux2Modulation.split}: a modulation
#' projection outputs \code{[N, 3 * n_sets * dim]}; this unsqueezes to
#' \code{[N, 1, ...]} and slices it into \code{n_sets} lists of three
#' \code{[N, 1, dim]} tensors.
#'
#' @param mod AnvlArray \code{[N, 3 * n_sets * dim]}.
#' @param n_sets Integer. Number of (shift, scale, gate) triples.
#' @param dim Integer. Model dimension.
#'
#' @return List of \code{n_sets} lists, each \code{list(shift, scale, gate)}.
#'
#' @keywords internal
.yq_mod_split <- function(mod, n_sets, dim) {
    s <- anvl::shape(mod)
    if (length(s) == 2L) {
        mod <- anvl::nv_reshape(mod, c(s[1L], 1L, s[2L]))
    }
    lapply(seq_len(n_sets), function(i) {
        base <- 3L * (i - 1L)
        lapply(seq_len(3L), function(j) {
            idx <- base + j
            yunque::yq_slice_lastdim(mod, (idx - 1L) * dim + 1L, idx * dim)
        })
    })
}

# Per-head RMS-normed projection: linear -> [B, S, H, D] -> rms(head).
.yq_qkv_head <- function(x, w_t, norm_w, heads, head_dim, eps) {
    s <- anvl::shape(x)
    proj <- yunque::yq_linear(x, w_t)
    r <- anvl::nv_reshape(proj, c(s[1L], s[2L], heads, head_dim))
    if (!is.null(norm_w)) {
        r <- yunque::yq_rms_norm(r, norm_w, eps = eps)
    }
    r
}

#' FLUX.2 double-stream (MMDiT) block (anvl port)
#'
#' anvl re-implementation of \code{diffuseR::flux2_double_block} +
#' \code{flux_attention(added_kv = TRUE)}: separate image and text
#' streams, each with parameterless-LayerNorm adaLN modulation and a
#' SwiGLU feed-forward, joined by a single attention over the
#' concatenated [text; image] sequence (text tokens first, matching the
#' rotary layout).
#'
#' Returns a closure over the static config; \code{anvl::jit()} the
#' closure. Weights travel as a named list (pytree) mirroring the
#' checkpoint keys under \code{transformer_blocks.N.}, all 2-D linears
#' pre-transposed to \code{[in, out]} (see
#' \code{\link{yq_read_safetensors}}).
#'
#' @param heads Integer. Attention heads (klein-4B: 24).
#' @param head_dim Integer. Per-head dimension (128).
#' @param mlp_ratio Numeric. Feed-forward multiplier (3.0).
#' @param eps Numeric. Norm epsilon.
#' @param precision Character. Matmul precision (see
#'   \code{\link[yunque]{yq_linear}}).
#'
#' @return Function of (h, c, mod_img, mod_txt, cos, sin, w):
#'   \itemize{
#'     \item h \code{[B, S_img, dim]} image hidden states
#'     \item c \code{[B, S_txt, dim]} text (encoder) hidden states
#'     \item mod_img, mod_txt \code{[B, 6*dim]} shared modulation outputs
#'     \item cos, sin \code{[S_txt + S_img, head_dim]} rotary tables for
#'       the concatenated sequence
#'     \item w named list of block weights (to_q/to_k/to_v, norm_q/k,
#'       add_{q,k,v}_proj, norm_added_{q,k}, to_out, to_add_out,
#'       ff.linear_{in,out}, ff_context.linear_{in,out})
#'   }
#'   returning \code{list(c_out, h_out)}.
#'
#' @export
yq_flux2_double_block <- function(heads = 24L, head_dim = 128L,
                                  mlp_ratio = 3.0, eps = 1e-6,
                                  precision = "highest") {
    heads <- as.integer(heads)
    head_dim <- as.integer(head_dim)
    inner <- heads * head_dim

    swiglu <- function(x, w_in_t, w_out_t) {
        proj <- yunque::yq_linear(x, w_in_t, precision = precision)
        half <- anvl::shape(proj)[3L] %/% 2L
        g <- yunque::yq_slice_lastdim(proj, 1L, half)
        u <- yunque::yq_slice_lastdim(proj, half + 1L, 2L * half)
        yunque::yq_linear(yunque::yq_silu(g) * u, w_out_t, precision = precision)
    }

    function(h, c, mod_img, mod_txt, cos, sin, w) {
        dim <- anvl::shape(h)[3L]
        s_txt <- anvl::shape(c)[2L]
        s_img <- anvl::shape(h)[2L]
        b <- anvl::shape(h)[1L]

        mi <- .yq_mod_split(mod_img, 2L, dim)
        mt <- .yq_mod_split(mod_txt, 2L, dim)
        msa <- mi[[1L]]; mmlp <- mi[[2L]]
        cmsa <- mt[[1L]]; cmlp <- mt[[2L]]

        hs <- anvl::shape(h)
        cs <- anvl::shape(c)
        modulate <- function(x, shift, scale) {
            sh <- anvl::shape(x)
            yunque::yq_layer_norm(x, eps = eps) *
            anvl::nv_broadcast_to(scale + 1, sh) +
            anvl::nv_broadcast_to(shift, sh)
        }

        norm_h <- modulate(h, msa[[1L]], msa[[2L]])
        norm_c <- modulate(c, cmsa[[1L]], cmsa[[2L]])

        # Joint attention. Per-head [B, S, H, D], text q/k/v first.
        q_i <- .yq_qkv_head(norm_h, w$to_q, w$norm_q, heads, head_dim, eps)
        k_i <- .yq_qkv_head(norm_h, w$to_k, w$norm_k, heads, head_dim, eps)
        v_i <- .yq_qkv_head(norm_h, w$to_v, NULL, heads, head_dim, eps)
        q_t <- .yq_qkv_head(norm_c, w$add_q_proj, w$norm_added_q,
                            heads, head_dim, eps)
        k_t <- .yq_qkv_head(norm_c, w$add_k_proj, w$norm_added_k,
                            heads, head_dim, eps)
        v_t <- .yq_qkv_head(norm_c, w$add_v_proj, NULL, heads, head_dim, eps)

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
        q <- yunque::yq_rope_apply(q, cb, sb)
        k <- yunque::yq_rope_apply(k, cb, sb)

        attn <- yunque::yq_sdpa(q, k, v, precision = precision)
        attn <- anvl::nv_reshape(anvl::nv_transpose(attn, perm),
                                 c(b, s_all, inner))

        ctx <- yunque::yq_slice_seq(attn, 1L, s_txt)
        img <- yunque::yq_slice_seq(attn, s_txt + 1L, s_all)
        attn_img <- yunque::yq_linear(img, w$to_out, precision = precision)
        attn_ctx <- yunque::yq_linear(ctx, w$to_add_out, precision = precision)

        h <- h + anvl::nv_broadcast_to(msa[[3L]], hs) * attn_img
        norm_h2 <- modulate(h, mmlp[[1L]], mmlp[[2L]])
        h <- h + anvl::nv_broadcast_to(mmlp[[3L]], hs) *
        swiglu(norm_h2, w$ff_in, w$ff_out)

        c <- c + anvl::nv_broadcast_to(cmsa[[3L]], cs) * attn_ctx
        norm_c2 <- modulate(c, cmlp[[1L]], cmlp[[2L]])
        c <- c + anvl::nv_broadcast_to(cmlp[[3L]], cs) *
        swiglu(norm_c2, w$ff_context_in, w$ff_context_out)

        list(c, h)
    }
}
