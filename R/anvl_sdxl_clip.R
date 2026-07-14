#' Stable Diffusion XL dual CLIP text encoders (anvl port)
#'
#' anvl re-implementation of the two SDXL conditioning encoders,
#' \code{diffuseR::text_encoder_native} (OpenAI CLIP ViT-L/14 text:
#' \code{embed_dim} 768, 12 layers, 12 heads, \code{head_dim} 64,
#' quick-GELU) and \code{diffuseR::text_encoder2_native} (OpenCLIP
#' ViT-bigG text: \code{embed_dim} 1280, 32 layers, 20 heads,
#' \code{head_dim} 64, exact/erf GELU). Both share the CLIP
#' text-transformer structure the SD 2.1 port uses (host-side token +
#' positional embedding gather, N pre-LayerNorm transformer layers with a
#' causal additive mask, a GELU MLP), parameterized here per encoder.
#'
#' \strong{Penultimate selection.} SDXL conditions on the
#' \emph{penultimate} hidden state of \emph{each} encoder
#' (\code{hidden_states[-2]} in diffusers: the output of the
#' second-to-last transformer layer, \strong{before} the final
#' LayerNorm), not the final-LN state SD 2.1 used. For an N-layer
#' encoder that is the hidden state after \code{N - 1} layers. The two
#' penultimate states are concatenated along the feature dim to
#' \code{[B, seq, 768 + 1280 = 2048]}, the UNet's
#' \code{cross_attention_dim}.
#'
#' \strong{Pooled embed.} bigG additionally produces the pooled vector the
#' UNet's text-time add-embedding consumes: run \emph{all} 32 layers,
#' apply the final LayerNorm, take the hidden state at the EOS token
#' position (argmax over the ids), and project through bigG's bias-free
#' \code{text_projection} to \code{[B, 1280]}.
#'
#' Per-encoder GELU: CLIP-L uses OpenAI quick-GELU
#' (\code{x * sigmoid(1.702 x)}); bigG uses the exact erf GELU (matching
#' \code{text_encoder2_native}'s \code{gelu_type = "exact"}). LayerNorm is
#' affine with eps 1e-5 (torch \code{nn_layer_norm} default).
#'
#' @name anvl_sdxl_clip
NULL

# CLIP LayerNorm epsilon (torch nn_layer_norm default).
.YQ_SDXL_CLIP_EPS <- 1e-5

# OpenAI quick-GELU (CLIP-L): x * sigmoid(1.702 * x). Scalars enter as
# ambiguous f32? and defer to x's f32 dtype (compute is f32 throughout).
.yq_sdxl_clip_quick_gelu <- function(x) {
    x * anvl::nv_logistic(x * 1.702)
}

# Exact GELU (erf form; bigG's gelu_type "exact", matching torch
# nnf_gelu default approximate="none"): 0.5 * x * (1 + erf(x / sqrt(2))).
.yq_sdxl_clip_exact_gelu <- function(x) {
    0.5 * x * (anvl::nv_erf(x * (1 / sqrt(2))) + 1)
}

# Affine LayerNorm over the last dim: yq_layer_norm (no affine) * w + b,
# with w/b [D] broadcast over [..., D]. (Named distinctly to avoid
# cross-file collisions.)
.yq_sdxl_clip_ln <- function(x, weight, bias, eps = .YQ_SDXL_CLIP_EPS) {
    s <- anvl::shape(x)
    yunque::layer_norm(x, eps = eps) *
        anvl::nv_broadcast_to(weight, s) + anvl::nv_broadcast_to(bias, s)
}

# One CLIP transformer layer: pre-norm multi-head self-attention (separate
# biased q/k/v/out projections, causal additive mask, scale 1/sqrt(D)) then
# pre-norm GELU MLP (fc1 -> gelu -> fc2), each residual. `gelu` selects the
# per-encoder activation.
.yq_sdxl_clip_layer <- function(n_heads, head_dim, gelu, eps, precision) {
    inner <- n_heads * head_dim

    function(x, mask, w) {
        s <- anvl::shape(x)
        b <- s[1L]
        n <- s[2L]

        h <- .yq_sdxl_clip_ln(x, w$ln1_w, w$ln1_b, eps)
        q <- yunque::linear(h, w$q_w, w$q_b, precision = precision)
        k <- yunque::linear(h, w$k_w, w$k_b, precision = precision)
        v <- yunque::linear(h, w$v_w, w$v_b, precision = precision)
        to_heads <- function(t) anvl::nv_transpose(
            anvl::nv_reshape(t, c(b, n, n_heads, head_dim)), c(1L, 3L, 2L, 4L))
        q <- to_heads(q)
        k <- to_heads(k)
        v <- to_heads(v)
        attn <- yunque::sdpa(q, k, v, mask = mask, precision = precision)
        attn <- anvl::nv_reshape(anvl::nv_transpose(attn, c(1L, 3L, 2L, 4L)),
                                 c(b, n, inner))
        x <- x + yunque::linear(attn, w$out_w, w$out_b, precision = precision)

        h2 <- .yq_sdxl_clip_ln(x, w$ln2_w, w$ln2_b, eps)
        mlp <- yunque::linear(
            gelu(yunque::linear(h2, w$fc1_w, w$fc1_b, precision = precision)),
            w$fc2_w, w$fc2_b, precision = precision)
        x + mlp
    }
}

#' SDXL CLIP causal additive attention mask (host-side)
#'
#' Upper-triangular causal mask as an additive bias \code{[B, 1, S, S]}
#' (0 where attended, a large negative where key j > query i). SDXL feeds
#' no padding mask (CLIP causal attention makes trailing pads irrelevant),
#' so this is causal-only. Broadcasts against scores inside
#' \code{\link[yunque]{yq_sdpa}}.
#'
#' @param seq_len Integer. Sequence length.
#' @param batch Integer. Batch size.
#' @param device Character. Target device.
#' @param neg Numeric. Masked-position bias.
#'
#' @return AnvlArray \code{[B, 1, S, S]}, f32.
#'
#' @export
yq_sdxl_clip_mask <- function(seq_len, batch = 1L, device = "cpu",
                              neg = -3.4e38) {
    causal <- matrix(0, seq_len, seq_len)
    causal[upper.tri(causal)] <- neg            # key j > query i masked
    arr <- array(0, dim = c(batch, 1L, seq_len, seq_len))
    for (b in seq_len(batch)) arr[b, 1L, , ] <- causal
    anvl::nv_array(arr, dtype = "f32", device = device)
}

#' SDXL CLIP token + positional embedding lookup (host-side gather)
#'
#' Gathers the token-embedding rows for the given ids and adds the learned
#' positional-embedding rows for positions \code{1..S}. Both tables stay R
#' matrices (never resident device tensors); only the summed
#' \code{[B, S, hidden]} result crosses to anvl. Mirrors the native
#' encoders' \code{token_embedding(ids + 1) + position_embedding[1:S]}.
#'
#' @param token_embedding R matrix \code{[vocab, hidden]} (from
#'   \code{\link{yq_sdxl_clip_load_weights}}).
#' @param position_embedding R matrix \code{[max_pos, hidden]}.
#' @param ids Integer matrix \code{[B, S]} of 0-based token ids.
#' @param device Character. Target device.
#'
#' @return AnvlArray \code{[B, S, hidden]}, f32.
#'
#' @export
yq_sdxl_clip_embed <- function(token_embedding, position_embedding, ids,
                               device = "cpu") {
    ids <- matrix(as.integer(ids), nrow = nrow(ids))
    b <- nrow(ids)
    s <- ncol(ids)
    hidden <- ncol(token_embedding)
    tok_rows <- token_embedding[as.integer(t(ids)) + 1L, , drop = FALSE]  # [B*S, hidden]
    pos_rows <- position_embedding[rep(seq_len(s), times = b), , drop = FALSE]
    combined <- tok_rows + pos_rows
    arr <- aperm(array(t(combined), dim = c(hidden, s, b)), c(3L, 2L, 1L))
    anvl::nv_array(arr, dtype = "f32", device = device)
}

#' SDXL CLIP-L (OpenAI ViT-L/14) penultimate encoder forward (anvl)
#'
#' Returns a closure over the static CLIP-L configuration;
#' \code{anvl::jit()} the closure. Runs the first \code{num_layers - 1}
#' transformer layers over pre-embedded tokens (quick-GELU MLP) and
#' returns that penultimate hidden state \code{[B, S, 768]}
#' (\code{hidden_states[-2]}, no final LayerNorm). The final layer's
#' weights load but are unused, matching diffusers' SDXL clip-skip=None
#' selection.
#'
#' @param num_layers Integer. Transformer layers (CLIP-L: 12).
#' @param num_heads Integer. Attention heads (12).
#' @param head_dim Integer. Per-head dimension (64).
#' @param eps Numeric. LayerNorm epsilon.
#' @param precision Character. Matmul precision.
#'
#' @return Function of \code{(embeds, mask, w)} returning \code{[B, S, 768]}.
#'
#' @export
yq_sdxl_clip_l_encoder <- function(num_layers = 12L, num_heads = 12L,
                                   head_dim = 64L, eps = 1e-5,
                                   precision = "highest") {
    num_layers <- as.integer(num_layers)
    layer <- .yq_sdxl_clip_layer(as.integer(num_heads), as.integer(head_dim),
                                 .yq_sdxl_clip_quick_gelu, eps, precision)

    function(embeds, mask, w) {
        x <- embeds
        for (i in seq_len(num_layers - 1L)) {
            x <- layer(x, mask, w$layers[[i]])
        }
        x
    }
}

#' SDXL OpenCLIP bigG penultimate + pooled encoder forward (anvl)
#'
#' Returns a closure over the static bigG configuration;
#' \code{anvl::jit()} the closure. Runs all \code{num_layers} transformer
#' layers over pre-embedded tokens (exact/erf GELU MLP), capturing the
#' penultimate hidden state (after \code{num_layers - 1} layers,
#' \code{hidden_states[-2]}) and applying the final LayerNorm to the full
#' output. Returns both so the caller can gather the pooled EOS row and
#' project it (see \code{\link{yq_sdxl_clip_encoders}}).
#'
#' @param num_layers Integer. Transformer layers (bigG: 32).
#' @param num_heads Integer. Attention heads (20).
#' @param head_dim Integer. Per-head dimension (64).
#' @param eps Numeric. LayerNorm epsilon.
#' @param precision Character. Matmul precision.
#'
#' @return Function of \code{(embeds, mask, w)} returning a list:
#'   \itemize{
#'     \item \code{penult} \code{[B, S, 1280]} penultimate hidden state
#'     \item \code{hidden_ln} \code{[B, S, 1280]} final-LN full hidden state
#'   }
#'
#' @export
yq_sdxl_bigg_encoder <- function(num_layers = 32L, num_heads = 20L,
                                 head_dim = 64L, eps = 1e-5,
                                 precision = "highest") {
    num_layers <- as.integer(num_layers)
    layer <- .yq_sdxl_clip_layer(as.integer(num_heads), as.integer(head_dim),
                                 .yq_sdxl_clip_exact_gelu, eps, precision)

    function(embeds, mask, w) {
        x <- embeds
        penult <- x
        for (i in seq_len(num_layers)) {
            if (i == num_layers) {
                penult <- x                 # hidden_states[-2]: state before the last layer
            }
            x <- layer(x, mask, w$layers[[i]])
        }
        hidden_ln <- .yq_sdxl_clip_ln(x, w$final_ln_w, w$final_ln_b, eps)
        list(penult = penult, hidden_ln = hidden_ln)
    }
}

#' Run both SDXL CLIP encoders (anvl)
#'
#' Runs the CLIP-L and OpenCLIP bigG encoders on pre-embedded tokens and
#' assembles the two SDXL conditioning tensors: the penultimate context
#' (concatenated CLIP-L + bigG penultimate states,
#' \code{[B, seq, 2048]}) and bigG's pooled text embed
#' (\code{[B, 1280]}). The two transformer stacks are \code{jit}-compiled;
#' the EOS gather and \code{text_projection} run eagerly.
#'
#' @param clipl_embeds AnvlArray \code{[B, S, 768]} CLIP-L token+pos
#'   embeddings (\code{\link{yq_sdxl_clip_embed}}).
#' @param bigg_embeds AnvlArray \code{[B, S, 1280]} bigG token+pos
#'   embeddings.
#' @param mask AnvlArray \code{[B, 1, S, S]} causal mask
#'   (\code{\link{yq_sdxl_clip_mask}}); shared by both encoders.
#' @param eos_index Integer. 1-based EOS position (\code{which.max} over
#'   the token ids). The pooled vector is gathered here. Batch = 1.
#' @param w_clipl,w_bigg Weight pytrees from
#'   \code{\link{yq_sdxl_clip_load_weights}}.
#' @param clipl_layers,clipl_heads,bigg_layers,bigg_heads,head_dim,eps
#'   Architecture (SDXL defaults).
#' @param precision Character. Matmul precision.
#' @param jit Logical. Compile the transformer stacks with
#'   \code{\link[anvl]{jit}}.
#'
#' @return List with \code{context} \code{[B, seq, 2048]} and
#'   \code{pooled} \code{[B, 1280]}.
#'
#' @export
yq_sdxl_clip_encoders <- function(clipl_embeds, bigg_embeds, mask, eos_index,
                                  w_clipl, w_bigg,
                                  clipl_layers = 12L, clipl_heads = 12L,
                                  bigg_layers = 32L, bigg_heads = 20L,
                                  head_dim = 64L, eps = 1e-5,
                                  precision = "highest", jit = TRUE) {
    fL <- yq_sdxl_clip_l_encoder(clipl_layers, clipl_heads, head_dim, eps,
                                 precision)
    fG <- yq_sdxl_bigg_encoder(bigg_layers, bigg_heads, head_dim, eps,
                               precision)
    if (jit) {
        fL <- anvl::jit(fL)
        fG <- anvl::jit(fG)
    }
    out_l <- fL(clipl_embeds, mask, w_clipl)                     # [B, S, 768]
    out_g <- fG(bigg_embeds, mask, w_bigg)                       # penult + hidden_ln
    context <- anvl::nv_concatenate(out_l, out_g$penult, dimension = 3L)

    # Pooled: EOS-position hidden state (final-LN) through text_projection.
    pooled_row <- anvl::nv_select(out_g$hidden_ln, 2L, as.integer(eos_index))
    pooled <- yunque::linear(pooled_row, w_bigg$text_projection,
                                precision = precision)

    list(context = context, pooled = pooled)
}

#' Load one SDXL CLIP text-encoder's weights into an anvl pytree
#'
#' Reads a \code{diffuseR::text_encoder_native} /
#' \code{text_encoder2_native} state_dict (F16/F32 upcast to f32) and
#' mirrors its key tree. The token- and positional-embedding tables stay R
#' matrices \code{[rows, hidden]} for host-side gather
#' (\code{\link{yq_sdxl_clip_embed}}); every other weight is an
#' \code{AnvlArray}, with attention / MLP / projection linears transposed
#' to \code{[in, out]} for \code{\link[yunque]{yq_linear}}. With
#' \code{strict = TRUE} every checkpoint key must be consumed exactly once.
#'
#' @param path Path to the encoder state_dict safetensors.
#' @param num_layers Integer. Transformer layers (CLIP-L: 12, bigG: 32).
#' @param has_text_projection Logical. bigG carries \code{text_projection}
#'   (pooled path); CLIP-L does not.
#' @param device Character. Target device.
#' @param strict Logical. Enforce the exact key census.
#'
#' @return Weights pytree for \code{\link{yq_sdxl_clip_l_encoder}} /
#'   \code{\link{yq_sdxl_bigg_encoder}} (plus \code{token_embedding} /
#'   \code{position_embedding} R matrices, and \code{text_projection} when
#'   present).
#'
#' @export
yq_sdxl_clip_load_weights <- function(path, num_layers,
                                      has_text_projection = FALSE,
                                      device = "cpu", strict = TRUE) {
    st <- yunque::st_open(path)
    on.exit(close(st$con))
    seen <- new.env(parent = emptyenv())
    mark <- function(key) assign(key, TRUE, envir = seen)
    raw <- function(key) {
        mark(key)
        anvl::nv_array(yunque::st_read(st, key), dtype = "f32",
                       device = device)
    }
    lin <- function(key) {
        mark(key)
        anvl::nv_array(yunque::st_read(st, key, transpose = TRUE),
                       dtype = "f32", device = device)
    }
    rmat <- function(key) {
        mark(key)
        yunque::st_read(st, key)              # R matrix [rows, hidden]
    }

    w <- list(
        token_embedding = rmat("token_embedding.weight"),
        position_embedding = rmat("position_embedding"),
        final_ln_w = raw("final_layer_norm.weight"),
        final_ln_b = raw("final_layer_norm.bias")
    )
    if (has_text_projection) {
        w$text_projection <- lin("text_projection.weight")
    }

    w$layers <- lapply(seq_len(num_layers) - 1L, function(i) {
        p <- sprintf("transformer_blocks.%d.", i)
        list(
            ln1_w = raw(paste0(p, "layernorm_1.weight")),
            ln1_b = raw(paste0(p, "layernorm_1.bias")),
            q_w = lin(paste0(p, "attention.q_proj.weight")),
            q_b = raw(paste0(p, "attention.q_proj.bias")),
            k_w = lin(paste0(p, "attention.k_proj.weight")),
            k_b = raw(paste0(p, "attention.k_proj.bias")),
            v_w = lin(paste0(p, "attention.v_proj.weight")),
            v_b = raw(paste0(p, "attention.v_proj.bias")),
            out_w = lin(paste0(p, "attention.out_proj.weight")),
            out_b = raw(paste0(p, "attention.out_proj.bias")),
            ln2_w = raw(paste0(p, "layernorm_2.weight")),
            ln2_b = raw(paste0(p, "layernorm_2.bias")),
            fc1_w = lin(paste0(p, "mlp.fc1.weight")),
            fc1_b = raw(paste0(p, "mlp.fc1.bias")),
            fc2_w = lin(paste0(p, "mlp.fc2.weight")),
            fc2_b = raw(paste0(p, "mlp.fc2.bias"))
        )
    })

    if (strict) {
        all_keys <- setdiff(names(st$header), "__metadata__")
        used <- ls(seen)
        unused <- setdiff(all_keys, used)
        extra <- setdiff(used, all_keys)
        if (length(extra)) {
            stop("SDXL CLIP anvl load: ", length(extra),
                 " requested keys not in file, e.g. ",
                 paste(utils::head(extra, 5), collapse = ", "))
        }
        if (length(unused)) {
            stop("SDXL CLIP anvl load: ", length(unused),
                 " checkpoint keys unused, e.g. ",
                 paste(utils::head(unused, 5), collapse = ", "))
        }
    }

    w
}
