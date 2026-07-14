#' Qwen3 rotary tables (host-side, Llama convention)
#'
#' \code{inv_freq = 1 / theta^(arange(0, D-2, 2) / D)}; cos/sin of the
#' position outer product. Returned as \code{[S, D/2]} AnvlArrays for the
#' split-half kernel (\code{\link[yunque]{yq_rope_split}}).
#'
#' @param seq_len Integer. Sequence length.
#' @param head_dim Integer. Per-head dim (128).
#' @param theta Numeric. RoPE base (Qwen3: 1e6).
#' @param device Character.
#'
#' @return List \code{list(cos, sin)}, each \code{[S, head_dim/2]}, f32.
#'
#' @export
yq_qwen3_rope <- function(seq_len, head_dim, theta = 1e6, device = "cpu") {
    r <- head_dim %/% 2L
    inv_freq <- 1 / theta^((2 * (0:(r - 1L))) / head_dim)
    pos <- 0:(seq_len - 1L)
    ang <- outer(pos, inv_freq)                # [S, r]
    list(cos = anvl::nv_array(cos(ang), dtype = "f32", device = device),
         sin = anvl::nv_array(sin(ang), dtype = "f32", device = device))
}

#' Qwen3 additive attention mask (host-side)
#'
#' Causal upper-triangular mask plus per-token padding, as an additive
#' bias \code{[B, 1, S, S]} (0 where attended, a large negative where
#' masked). Broadcasts against scores inside \code{\link[yunque]{yq_sdpa}}.
#'
#' @param attention_mask Integer/numeric matrix \code{[B, S]} (1 real,
#'   0 pad), or NULL for causal-only.
#' @param seq_len Integer. Sequence length (rows of attention_mask).
#' @param batch Integer. Batch size.
#' @param device Character.
#' @param neg Numeric. Masked-position bias.
#'
#' @return AnvlArray \code{[B, 1, S, S]}, f32.
#'
#' @export
yq_qwen3_mask <- function(attention_mask, seq_len, batch = 1L,
                          device = "cpu", neg = -3.4e38) {
    causal <- matrix(0, seq_len, seq_len)
    causal[upper.tri(causal)] <- neg           # key j > query i masked
    arr <- array(0, dim = c(batch, 1L, seq_len, seq_len))
    for (b in seq_len(batch)) {
        m <- causal
        if (!is.null(attention_mask)) {
            pad <- (1 - attention_mask[b, ]) * neg     # [S] over keys
            m <- m + matrix(pad, seq_len, seq_len, byrow = TRUE)
        }
        arr[b, 1L, , ] <- m
    }
    anvl::nv_array(arr, dtype = "f32", device = device)
}

#' Qwen3 token embedding lookup (host-side gather)
#'
#' Gathers rows of the embedding table for the given token ids. The
#' table stays an R matrix (never a resident device tensor); the result
#' is the only thing that crosses to anvl.
#'
#' @param embed R matrix \code{[vocab, hidden]} (from
#'   \code{\link{yq_qwen3_load_weights}}).
#' @param ids Integer matrix \code{[B, S]} of 0-based token ids.
#' @param device Character.
#'
#' @return AnvlArray \code{[B, S, hidden]}, f32.
#'
#' @export
yq_qwen3_embed <- function(embed, ids, device = "cpu") {
    ids <- matrix(as.integer(ids), nrow = nrow(ids))
    b <- nrow(ids); s <- ncol(ids); hidden <- ncol(embed)
    rows <- embed[as.integer(t(ids)) + 1L, , drop = FALSE]  # [B*S, hidden], row-major
    arr <- aperm(array(t(rows), dim = c(hidden, s, b)), c(3L, 2L, 1L))
    anvl::nv_array(arr, dtype = "f32", device = device)
}

# One Qwen3 decoder layer: pre-norm attention (GQA, per-head q/k RMS
# norm, split RoPE, additive mask) + pre-norm SwiGLU MLP, residual.
.yq_qwen3_layer <- function(num_heads, num_kv, head_dim, eps, precision) {
    inner <- num_heads * head_dim
    kv_inner <- num_kv * head_dim
    groups <- num_heads %/% num_kv
    r <- head_dim %/% 2L

    function(x, cos, sin, mask, w) {
        s <- anvl::shape(x)
        b <- s[1L]; n <- s[2L]

        h <- yunque::rms_norm(x, w$in_ln, eps = eps)
        q <- anvl::nv_reshape(yunque::linear(h, w$q_proj, precision = precision),
                              c(b, n, num_heads, head_dim))
        k <- anvl::nv_reshape(yunque::linear(h, w$k_proj, precision = precision),
                              c(b, n, num_kv, head_dim))
        v <- anvl::nv_reshape(yunque::linear(h, w$v_proj, precision = precision),
                              c(b, n, num_kv, head_dim))
        q <- yunque::rms_norm(q, w$q_norm, eps = eps)
        k <- yunque::rms_norm(k, w$k_norm, eps = eps)

        perm <- c(1L, 3L, 2L, 4L)              # [B, S, H, D] -> [B, H, S, D]
        q <- anvl::nv_transpose(q, perm)
        k <- anvl::nv_transpose(k, perm)
        v <- anvl::nv_transpose(v, perm)

        cq <- anvl::nv_broadcast_to(anvl::nv_reshape(cos, c(1L, 1L, n, r)),
                                    c(b, num_heads, n, r))
        sq <- anvl::nv_broadcast_to(anvl::nv_reshape(sin, c(1L, 1L, n, r)),
                                    c(b, num_heads, n, r))
        ck <- anvl::nv_broadcast_to(anvl::nv_reshape(cos, c(1L, 1L, n, r)),
                                    c(b, num_kv, n, r))
        sk <- anvl::nv_broadcast_to(anvl::nv_reshape(sin, c(1L, 1L, n, r)),
                                    c(b, num_kv, n, r))
        q <- yunque::rope_split(q, cq, sq)
        k <- yunque::rope_split(k, ck, sk)

        k <- yunque::repeat_kv(k, groups)
        v <- yunque::repeat_kv(v, groups)

        attn <- yunque::sdpa(q, k, v, mask = mask, precision = precision)
        attn <- anvl::nv_reshape(anvl::nv_transpose(attn, perm),
                                 c(b, n, inner))
        x <- x + yunque::linear(attn, w$o_proj, precision = precision)

        h2 <- yunque::rms_norm(x, w$post_ln, eps = eps)
        mlp <- yunque::linear(
            yunque::silu(yunque::linear(h2, w$gate, precision = precision)) *
            yunque::linear(h2, w$up, precision = precision),
            w$down, precision = precision)
        x + mlp
    }
}

#' Qwen3 encoder forward for FLUX.2 (anvl port)
#'
#' anvl re-implementation of \code{diffuseR::qwen3_encoder}, restricted
#' to what FLUX.2 klein needs: runs to \code{max(out_layers)} decoder
#' layers over pre-embedded tokens and returns the requested mid-stack
#' hidden states concatenated per token (no final norm, no LM head).
#' Defaults are Qwen3-4B.
#'
#' @param out_layers Integer vector. 1-based layer depths whose outputs
#'   are concatenated (klein-4B: 9, 18, 27).
#' @param num_heads,num_kv,head_dim Integers. Attention shape.
#' @param eps Numeric. RMS norm epsilon.
#' @param precision Character. Matmul precision.
#'
#' @return Function of (embeds, cos, sin, mask, w):
#'   \itemize{
#'     \item embeds \code{[B, S, hidden]} from \code{\link{yq_qwen3_embed}}
#'     \item cos, sin \code{[S, head_dim/2]} from \code{\link{yq_qwen3_rope}}
#'     \item mask \code{[B, 1, S, S]} from \code{\link{yq_qwen3_mask}}
#'     \item w weights pytree (\code{w$layers[[i]]})
#'   }
#'   returning \code{[B, S, length(out_layers) * hidden]}.
#'
#' @export
yq_qwen3_encoder <- function(out_layers = c(9L, 18L, 27L),
                             num_heads = 32L, num_kv = 8L, head_dim = 128L,
                             eps = 1e-6, precision = "highest") {
    out_layers <- sort(as.integer(out_layers))
    n_run <- max(out_layers)
    layer <- .yq_qwen3_layer(num_heads, num_kv, head_dim, eps, precision)

    function(embeds, cos, sin, mask, w) {
        x <- embeds
        states <- vector("list", length(out_layers))
        for (i in seq_len(n_run)) {
            x <- layer(x, cos, sin, mask, w$layers[[i]])
            hit <- which(out_layers == i)
            if (length(hit)) states[[hit]] <- x
        }
        do.call(anvl::nv_concatenate,
                c(states, list(dimension = anvl::ndims(x))))
    }
}
