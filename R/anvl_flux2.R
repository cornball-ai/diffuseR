#' FLUX.2 single-stream block (anvl port)
#'
#' anvl re-implementation of \code{diffuseR::flux2_single_block} +
#' \code{flux2_parallel_self_attention}: parameterless LayerNorm with
#' external (shift, scale, gate) modulation, one fused projection
#' producing QKV and the SwiGLU MLP input, RMS-normed q/k, rotary
#' embeddings, attention, and one fused output projection consuming
#' cat(attention, MLP).
#'
#' Returns a closure over the static configuration; \code{anvl::jit()}
#' the closure. All array arguments are dynamic inputs.
#'
#' @param heads Integer. Attention heads (klein-4B: 24).
#' @param head_dim Integer. Per-head dimension (128).
#' @param mlp_ratio Numeric. MLP hidden multiplier (3.0).
#' @param eps Numeric. Norm epsilon.
#' @param precision Character. Matmul precision for every linear and
#'   attention matmul in the block (see \code{\link[yunque]{yq_linear}}).
#'
#' @return Function of (h, shift, scale, gate, cos, sin, w_qkv_t,
#'   w_out_t, norm_q_w, norm_k_w):
#'   \itemize{
#'     \item h \code{[B, S, dim]} joint (text; image) hidden states
#'     \item shift, scale, gate \code{[1, 1, dim]} modulation triple
#'     \item cos, sin \code{[S, head_dim]} rotary tables
#'     \item w_qkv_t \code{[dim, 3*heads*head_dim + 2*mlp_hidden]}
#'       (checkpoint weight transposed)
#'     \item w_out_t \code{[heads*head_dim + mlp_hidden, dim]}
#'     \item norm_q_w, norm_k_w \code{[head_dim]}
#'   }
#'
#' @export
yq_flux2_single_block <- function(heads = 24L, head_dim = 128L,
                                  mlp_ratio = 3.0, eps = 1e-6,
                                  precision = "highest") {
    heads <- as.integer(heads)
    head_dim <- as.integer(head_dim)
    inner <- heads * head_dim

    function(h, shift, scale, gate, cos, sin,
             w_qkv_t, w_out_t, norm_q_w, norm_k_w) {
        s <- anvl::shape(h)
        b <- s[1L]
        n <- s[2L]
        dim <- s[3L]
        mlp_hidden <- as.integer(dim * mlp_ratio)

        norm_h <- yunque::yq_layer_norm(h, eps = eps)
        norm_h <- norm_h * anvl::nv_broadcast_to(scale + 1, s) +
        anvl::nv_broadcast_to(shift, s)

        proj <- yunque::yq_linear(norm_h, w_qkv_t, precision = precision)
        q <- yunque::yq_slice_lastdim(proj, 1L, inner)
        k <- yunque::yq_slice_lastdim(proj, inner + 1L, 2L * inner)
        v <- yunque::yq_slice_lastdim(proj, 2L * inner + 1L, 3L * inner)
        m1 <- yunque::yq_slice_lastdim(proj, 3L * inner + 1L, 3L * inner + mlp_hidden)
        m2 <- yunque::yq_slice_lastdim(proj, 3L * inner + mlp_hidden + 1L,
                               3L * inner + 2L * mlp_hidden)

        # [B, S, inner] -> [B, S, H, D], q/k norms on the head dim,
        # then -> [B, H, S, D] (same order as the torch reference)
        q <- anvl::nv_reshape(q, c(b, n, heads, head_dim))
        k <- anvl::nv_reshape(k, c(b, n, heads, head_dim))
        v <- anvl::nv_reshape(v, c(b, n, heads, head_dim))
        q <- yunque::yq_rms_norm(q, norm_q_w, eps = eps)
        k <- yunque::yq_rms_norm(k, norm_k_w, eps = eps)
        perm <- c(1L, 3L, 2L, 4L)
        q <- anvl::nv_transpose(q, perm)
        k <- anvl::nv_transpose(k, perm)
        v <- anvl::nv_transpose(v, perm)

        hs <- c(b, heads, n, head_dim)
        cs <- anvl::nv_broadcast_to(cos, hs)
        sn <- anvl::nv_broadcast_to(sin, hs)
        q <- yunque::yq_rope_apply(q, cs, sn)
        k <- yunque::yq_rope_apply(k, cs, sn)

        attn <- yunque::yq_sdpa(q, k, v, precision = precision)
        attn <- anvl::nv_reshape(anvl::nv_transpose(attn, perm),
                                 c(b, n, inner))

        mlp <- yunque::yq_silu(m1) * m2
        out <- yunque::yq_linear(anvl::nv_concatenate(attn, mlp, dimension = 3L),
                         w_out_t, precision = precision)
        h + anvl::nv_broadcast_to(gate, s) * out
    }
}
