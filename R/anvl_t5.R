#' T5-XXL encoder forward (anvl port of \code{\link{t5_encoder}})
#'
#' anvl re-implementation of diffuseR's T5-v1.1 encoder stack, FLUX.1's
#' second text encoder. Mirrors the torch reference module tree 1:1 but
#' as pure functions over an AnvlArray weight pytree. T5 distinctives,
#' each carried over exactly:
#' \itemize{
#'   \item \strong{RMS norm} (\code{\link[yunque]{yq_rms_norm}}: scale
#'     only, no bias, no mean subtraction, no +1 on the weight).
#'   \item \strong{No attention scaling} - scores are the raw
#'     \eqn{q k^\top}, never divided by \eqn{\sqrt{d_{kv}}} (folded into
#'     the weights), so \code{\link[yunque]{yq_sdpa}} (which scales) is
#'     NOT used; scores are formed manually.
#'   \item \strong{Learned relative-position bias} added to the unscaled
#'     scores pre-softmax; computed once (host-side,
#'     \code{\link{yq_t5_rel_pos_bias}}) from block 1's bucketed embedding
#'     and shared by every layer.
#'   \item \strong{Gated-GELU (GEGLU) FFN}: \code{wo(gelu(wi_0(x)) *
#'     wi_1(x))} with the tanh-approximation GELU (T5 v1.1 = "gelu_new").
#' }
#' The encoder is bidirectional (padding mask only); FLUX passes no mask,
#' so none is implemented here.
#'
#' @name anvl_t5
NULL

# Batched matmul honouring anvl's optional strict-f32 precision arg.
.yq_t5_matmul <- function(lhs, rhs, precision = "highest") {
    if ("precision" %in% names(formals(anvl::nv_matmul))) {
        anvl::nv_matmul(lhs, rhs, precision = precision)
    } else {
        anvl::nv_matmul(lhs, rhs)
    }
}

# tanh-approximation GELU (torch nnf_gelu(approximate = "tanh") /
# transformers "gelu_new"): 0.5 x (1 + tanh(sqrt(2/pi)(x + 0.044715 x^3))).
.yq_t5_gelu_tanh <- function(z) {
    inner <- (z + (z * z * z) * 0.044715) * sqrt(2 / pi)
    (z * 0.5) * (anvl::nv_tanh(inner) + 1)
}

# One T5 encoder block: pre-norm self-attention (no scale, additive
# relative-position bias) + residual, then pre-norm gated-GELU FFN +
# residual. position_bias is the shared [1, H, S, S] tensor.
.yq_t5_layer <- function(num_heads, d_kv, eps, precision) {
    inner <- num_heads * d_kv

    function(x, position_bias, w) {
        s <- anvl::shape(x)
        b <- s[1L]; n <- s[2L]
        per <- c(b, n, num_heads, d_kv)
        perm <- c(1L, 3L, 2L, 4L)              # [B, S, H, D] <-> [B, H, S, D]

        # --- self-attention (pre-norm) ---
        h <- yunque::yq_rms_norm(x, w$attn_ln, eps = eps)
        q <- anvl::nv_transpose(
            anvl::nv_reshape(yunque::yq_linear(h, w$q, precision = precision), per),
            perm)
        k <- anvl::nv_transpose(
            anvl::nv_reshape(yunque::yq_linear(h, w$k, precision = precision), per),
            perm)
        v <- anvl::nv_transpose(
            anvl::nv_reshape(yunque::yq_linear(h, w$v, precision = precision), per),
            perm)

        # scores = q k^T with NO 1/sqrt(d) scaling; add shared rel-pos bias
        scores <- .yq_t5_matmul(q, anvl::nv_transpose(k, c(1L, 2L, 4L, 3L)),
                                precision = precision)
        scores <- scores + anvl::nv_broadcast_to(position_bias, anvl::shape(scores))
        attn <- yunque::yq_softmax(scores)
        out <- .yq_t5_matmul(attn, v, precision = precision)
        out <- anvl::nv_reshape(anvl::nv_transpose(out, perm), c(b, n, inner))
        x <- x + yunque::yq_linear(out, w$o, precision = precision)

        # --- gated-GELU feed-forward (pre-norm) ---
        h2 <- yunque::yq_rms_norm(x, w$ff_ln, eps = eps)
        gated <- .yq_t5_gelu_tanh(yunque::yq_linear(h2, w$wi_0, precision = precision)) *
            yunque::yq_linear(h2, w$wi_1, precision = precision)
        x + yunque::yq_linear(gated, w$wo, precision = precision)
    }
}

#' T5 relative-position bias (host-side, bucketed)
#'
#' Reproduces \code{T5Attention.compute_bias}: bucket the bidirectional
#' relative positions \eqn{j - i} (half the buckets split by sign, half
#' of each side exact small offsets, the rest log-spaced up to
#' \code{max_distance}), gather block 1's learned bias embedding, and lay
#' it out as the additive \code{[1, H, S, S]} tensor added to every
#' layer's attention logits. Parameter-free integer indexing, so it is
#' computed in base R and passed into the jitted encoder as an input
#' (same boundary logic as the token embedding gather).
#'
#' @param rel_bias_weight R matrix \code{[num_buckets, num_heads]} (the
#'   \code{relative_attention_bias} embedding, from
#'   \code{\link{yq_t5_load_weights}}).
#' @param seq_len Integer. Sequence length.
#' @param num_buckets Integer. Total relative-position buckets (32).
#' @param max_distance Integer. Max distance before log-spacing saturates
#'   (128).
#' @param device Character.
#'
#' @return AnvlArray \code{[1, num_heads, seq_len, seq_len]}, f32.
#'
#' @export
yq_t5_rel_pos_bias <- function(rel_bias_weight, seq_len, num_buckets = 32L,
                               max_distance = 128L, device = "cpu") {
    S <- as.integer(seq_len)
    H <- ncol(rel_bias_weight)
    pos <- 0:(S - 1L)
    rp <- outer(pos, pos, function(qi, kj) kj - qi)   # rp[i, j] = j - i

    nb <- num_buckets %/% 2L                           # bidirectional halving
    rel_buckets <- (rp > 0) * nb                       # 0 or nb by sign
    rpa <- abs(rp)
    max_exact <- nb %/% 2L
    is_small <- rpa < max_exact
    # log-spaced large bucket; guard the log arg so unused (small) cells
    # do not produce -Inf/NA (ifelse discards them anyway).
    rpa_safe <- pmax(rpa, max_exact)
    rp_large <- as.integer(log(rpa_safe / max_exact) /
                           log(max_distance / max_exact) * (nb - max_exact)) +
        max_exact                                      # trunc toward 0 == torch long
    rp_large <- pmin(rp_large, nb - 1L)
    bucket <- rel_buckets + ifelse(is_small, rpa, rp_large)   # 0-based [S, S]

    idx <- as.integer(bucket) + 1L                     # col-major flatten
    gathered <- rel_bias_weight[idx, , drop = FALSE]   # [S*S, H]
    arr <- aperm(array(gathered, dim = c(S, S, H)), c(3L, 1L, 2L))  # [H, i, j]
    dim(arr) <- c(1L, H, S, S)                         # [1, H, i, j]
    anvl::nv_array(arr, dtype = "f32", device = device)
}

#' T5 token embedding lookup (host-side gather)
#'
#' Gathers rows of the shared embedding table for the given token ids.
#' The table stays an R matrix (never resident on device); only the
#' gathered \code{[B, S, d_model]} result crosses to anvl.
#'
#' @param embed R matrix \code{[vocab, d_model]} (from
#'   \code{\link{yq_t5_load_weights}}).
#' @param ids Integer matrix \code{[B, S]} of 0-based token ids.
#' @param device Character.
#'
#' @return AnvlArray \code{[B, S, d_model]}, f32.
#'
#' @export
yq_t5_embed <- function(embed, ids, device = "cpu") {
    ids <- matrix(as.integer(ids), nrow = nrow(ids))
    b <- nrow(ids); s <- ncol(ids); hidden <- ncol(embed)
    rows <- embed[as.integer(t(ids)) + 1L, , drop = FALSE]  # [B*S, hidden]
    arr <- aperm(array(t(rows), dim = c(hidden, s, b)), c(3L, 2L, 1L))
    anvl::nv_array(arr, dtype = "f32", device = device)
}

#' Load T5 encoder weights into an anvl pytree
#'
#' Reads a T5 state_dict from a safetensors file (F32/F16/BF16), keys
#' mirroring \code{\link{t5_encoder}}'s module tree. Linear weights are
#' transposed to \code{[in, out]} for \code{\link[yunque]{yq_linear}};
#' the shared embedding and the relative-position bias embedding stay R
#' matrices for host-side gather. Strict census: every requested key must
#' exist.
#'
#' @param path safetensors file with the T5 encoder state_dict.
#' @param num_layers Integer. Encoder blocks to load.
#' @param device Character. Target device.
#'
#' @return List \code{list(embed = <[vocab, d_model] R matrix>,
#'   rel_bias = <[num_buckets, num_heads] R matrix>, layers = <per-layer
#'   weight lists>, final_ln = <AnvlArray>)}.
#'
#' @export
yq_t5_load_weights <- function(path, num_layers = 24L, device = "cpu") {
    st <- yunque::yq_st_open(path)
    on.exit(close(st$con))
    lin <- function(key) anvl::nv_array(yunque::yq_st_read(st, key, transpose = TRUE),
                                        dtype = "f32", device = device)
    vec <- function(key) anvl::nv_array(yunque::yq_st_read(st, key),
                                        dtype = "f32", device = device)

    embed <- yunque::yq_st_read(st, "shared.weight")           # [vocab, d_model]
    rel_bias <- yunque::yq_st_read(st,
        "block.0.layer.0.SelfAttention.relative_attention_bias.weight")  # [buckets, heads]

    layers <- lapply(seq_len(num_layers) - 1L, function(i) {
        p <- sprintf("block.%d.", i)
        list(
            attn_ln = vec(paste0(p, "layer.0.layer_norm.weight")),
            q = lin(paste0(p, "layer.0.SelfAttention.q.weight")),
            k = lin(paste0(p, "layer.0.SelfAttention.k.weight")),
            v = lin(paste0(p, "layer.0.SelfAttention.v.weight")),
            o = lin(paste0(p, "layer.0.SelfAttention.o.weight")),
            ff_ln = vec(paste0(p, "layer.1.layer_norm.weight")),
            wi_0 = lin(paste0(p, "layer.1.DenseReluDense.wi_0.weight")),
            wi_1 = lin(paste0(p, "layer.1.DenseReluDense.wi_1.weight")),
            wo = lin(paste0(p, "layer.1.DenseReluDense.wo.weight"))
        )
    })

    list(embed = embed, rel_bias = rel_bias, layers = layers,
         final_ln = vec("final_layer_norm.weight"))
}

#' T5-XXL encoder forward (jit-ready closure)
#'
#' Runs \code{num_layers} T5 encoder blocks over pre-embedded tokens and
#' applies the final RMS norm, returning the last hidden state. Defaults
#' are the T5-v1.1-XXL configuration used by FLUX.1.
#'
#' @param num_layers Integer. Encoder blocks (XXL: 24).
#' @param num_heads,d_kv Integers. Attention shape (XXL: 64 heads x 64).
#' @param eps Numeric. RMS norm epsilon.
#' @param precision Character. Matmul precision.
#'
#' @return Function of \code{(embeds, position_bias, w)}:
#'   \itemize{
#'     \item embeds \code{[B, S, d_model]} from \code{\link{yq_t5_embed}}
#'     \item position_bias \code{[1, H, S, S]} from
#'       \code{\link{yq_t5_rel_pos_bias}}
#'     \item w weights pytree from \code{\link{yq_t5_load_weights}}
#'   }
#'   returning \code{[B, S, d_model]}.
#'
#' @export
yq_t5_encoder <- function(num_layers = 24L, num_heads = 64L, d_kv = 64L,
                          eps = 1e-6, precision = "highest") {
    layer <- .yq_t5_layer(num_heads, d_kv, eps, precision)

    function(embeds, position_bias, w) {
        x <- embeds
        for (i in seq_len(num_layers)) {
            x <- layer(x, position_bias, w$layers[[i]])
        }
        yunque::yq_rms_norm(x, w$final_ln, eps = eps)
    }
}
