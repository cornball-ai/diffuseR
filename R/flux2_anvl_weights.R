#' Load FLUX.2 Klein transformer weights into an anvl pytree
#'
#' Reads every transformer weight from the checkpoint (bf16 upcast to
#' f32), transposing 2-D linears to \code{[in, out]}, and wraps each as
#' an \code{AnvlArray} on \code{device} — freeing the R copy as it goes,
#' so peak host memory stays near one tensor rather than the full 15.5
#' GB twice. Returns the nested list \code{\link{yq_flux2_transformer}}
#' expects.
#'
#' @param path Path to \code{transformer/diffusion_pytorch_model.safetensors}.
#' @param num_layers Integer. Double-stream blocks (5).
#' @param num_single_layers Integer. Single-stream blocks (20).
#' @param device Character. Target device.
#'
#' @return Weights pytree.
#'
#' @export
yq_flux2_load_weights <- function(path, num_layers = 5L,
                                  num_single_layers = 20L, device = "cpu") {
    st <- yunque::st_open(path)
    on.exit(close(st$con))
    lin <- function(key) {
        a <- anvl::nv_array(yunque::st_read(st, key, transpose = TRUE),
                            dtype = "f32", device = device)
        a
    }
    vec <- function(key) {
        anvl::nv_array(yunque::st_read(st, key), dtype = "f32", device = device)
    }

    w <- list(
        x_embedder = lin("x_embedder.weight"),
        context_embedder = lin("context_embedder.weight"),
        time_1 = lin("time_guidance_embed.timestep_embedder.linear_1.weight"),
        time_2 = lin("time_guidance_embed.timestep_embedder.linear_2.weight"),
        dsm_img = lin("double_stream_modulation_img.linear.weight"),
        dsm_txt = lin("double_stream_modulation_txt.linear.weight"),
        single_mod = lin("single_stream_modulation.linear.weight"),
        norm_out = lin("norm_out.linear.weight"),
        proj_out = lin("proj_out.weight")
    )

    w$double <- lapply(seq_len(num_layers) - 1L, function(i) {
        p <- sprintf("transformer_blocks.%d.", i)
        list(
            to_q = lin(paste0(p, "attn.to_q.weight")),
            to_k = lin(paste0(p, "attn.to_k.weight")),
            to_v = lin(paste0(p, "attn.to_v.weight")),
            norm_q = vec(paste0(p, "attn.norm_q.weight")),
            norm_k = vec(paste0(p, "attn.norm_k.weight")),
            add_q_proj = lin(paste0(p, "attn.add_q_proj.weight")),
            add_k_proj = lin(paste0(p, "attn.add_k_proj.weight")),
            add_v_proj = lin(paste0(p, "attn.add_v_proj.weight")),
            norm_added_q = vec(paste0(p, "attn.norm_added_q.weight")),
            norm_added_k = vec(paste0(p, "attn.norm_added_k.weight")),
            to_out = lin(paste0(p, "attn.to_out.0.weight")),
            to_add_out = lin(paste0(p, "attn.to_add_out.weight")),
            ff_in = lin(paste0(p, "ff.linear_in.weight")),
            ff_out = lin(paste0(p, "ff.linear_out.weight")),
            ff_context_in = lin(paste0(p, "ff_context.linear_in.weight")),
            ff_context_out = lin(paste0(p, "ff_context.linear_out.weight"))
        )
    })

    w$single <- lapply(seq_len(num_single_layers) - 1L, function(i) {
        p <- sprintf("single_transformer_blocks.%d.", i)
        list(
            qkv = lin(paste0(p, "attn.to_qkv_mlp_proj.weight")),
            out = lin(paste0(p, "attn.to_out.weight")),
            norm_q = vec(paste0(p, "attn.norm_q.weight")),
            norm_k = vec(paste0(p, "attn.norm_k.weight"))
        )
    })

    w
}

#' Load Qwen3-4B text-encoder weights for FLUX.2 into an anvl pytree
#'
#' Reads the sharded \code{text_encoder} checkpoint (bf16 upcast to f32).
#' The embedding table stays an R matrix for host-side gather
#' (\code{\link{yq_qwen3_embed}}); only the first \code{n_layers} decoder
#' layers are loaded (klein consumes mid-stack states, so the deeper
#' layers and the tied LM head are never needed). Each tensor is wrapped
#' as an \code{AnvlArray} as it is read, freeing the R copy.
#'
#' @param dir The \code{text_encoder} directory (index + shards).
#' @param n_layers Integer. Decoder layers to load (klein: 27, enough
#'   for out_layers up to 27).
#' @param device Character. Target device.
#'
#' @return List \code{list(embed = <R matrix [vocab, hidden]>,
#'   layers = <list of per-layer weight lists>)}.
#'
#' @export
yq_qwen3_load_weights <- function(dir, n_layers = 27L, device = "cpu") {
    st <- yunque::st_open_sharded(dir)
    on.exit(yunque::st_close(st))
    lin <- function(key) anvl::nv_array(yunque::st_read(st, key, transpose = TRUE),
                                        dtype = "f32", device = device)
    vec <- function(key) anvl::nv_array(yunque::st_read(st, key),
                                        dtype = "f32", device = device)

    embed <- yunque::st_read(st, "model.embed_tokens.weight")   # [vocab, hidden]

    layers <- lapply(seq_len(n_layers) - 1L, function(i) {
        p <- sprintf("model.layers.%d.", i)
        list(
            in_ln = vec(paste0(p, "input_layernorm.weight")),
            post_ln = vec(paste0(p, "post_attention_layernorm.weight")),
            q_proj = lin(paste0(p, "self_attn.q_proj.weight")),
            k_proj = lin(paste0(p, "self_attn.k_proj.weight")),
            v_proj = lin(paste0(p, "self_attn.v_proj.weight")),
            o_proj = lin(paste0(p, "self_attn.o_proj.weight")),
            q_norm = vec(paste0(p, "self_attn.q_norm.weight")),
            k_norm = vec(paste0(p, "self_attn.k_norm.weight")),
            gate = lin(paste0(p, "mlp.gate_proj.weight")),
            up = lin(paste0(p, "mlp.up_proj.weight")),
            down = lin(paste0(p, "mlp.down_proj.weight"))
        )
    })

    list(embed = embed, layers = layers)
}

#' Load FLUX.2 VAE decoder weights into an anvl pytree
#'
#' Reads the decoder half of the \code{AutoencoderKLFlux2} checkpoint
#' (bf16 upcast to f32): \code{post_quant_conv}, the decoder body, the
#' output head, and the BatchNorm latent statistics. Conv weights are
#' loaded in torch \code{[out, in, kH, kW]} layout (what
#' \code{nv_conv2d} expects); the attention linears are transposed to
#' \code{[in, out]} for \code{yq_linear}. Encoder / quant_conv keys are
#' skipped (txt2img needs only the decode path).
#'
#' @param path Path to \code{vae/diffusion_pytorch_model.safetensors}.
#' @param device Character. Target device.
#'
#' @return VAE weights pytree; \code{bn_mean}/\code{bn_var} are returned
#'   as R vectors for host-side de-normalization
#'   (\code{\link{yq_flux2_vae_prepare}}).
#'
#' @export
yq_flux2_load_vae <- function(path, device = "cpu") {
    st <- yunque::st_open(path)
    on.exit(close(st$con))
    raw <- function(key) anvl::nv_array(yunque::st_read(st, key),
                                        dtype = "f32", device = device)
    lin <- function(key) anvl::nv_array(yunque::st_read(st, key, transpose = TRUE),
                                        dtype = "f32", device = device)
    has <- function(key) !is.null(st$header[[key]])

    resnet <- function(p) {
        r <- list(
            norm1_w = raw(paste0(p, "norm1.weight")),
            norm1_b = raw(paste0(p, "norm1.bias")),
            conv1_w = raw(paste0(p, "conv1.weight")),
            conv1_b = raw(paste0(p, "conv1.bias")),
            norm2_w = raw(paste0(p, "norm2.weight")),
            norm2_b = raw(paste0(p, "norm2.bias")),
            conv2_w = raw(paste0(p, "conv2.weight")),
            conv2_b = raw(paste0(p, "conv2.bias"))
        )
        if (has(paste0(p, "conv_shortcut.weight"))) {
            r$shortcut_w <- raw(paste0(p, "conv_shortcut.weight"))
            r$shortcut_b <- raw(paste0(p, "conv_shortcut.bias"))
        }
        r
    }

    dp <- "decoder."
    w <- list(
        post_quant_w = raw("post_quant_conv.weight"),
        post_quant_b = raw("post_quant_conv.bias"),
        conv_in_w = raw(paste0(dp, "conv_in.weight")),
        conv_in_b = raw(paste0(dp, "conv_in.bias")),
        norm_out_w = raw(paste0(dp, "conv_norm_out.weight")),
        norm_out_b = raw(paste0(dp, "conv_norm_out.bias")),
        conv_out_w = raw(paste0(dp, "conv_out.weight")),
        conv_out_b = raw(paste0(dp, "conv_out.bias")),
        bn_mean = yunque::st_read(st, "bn.running_mean"),
        bn_var = yunque::st_read(st, "bn.running_var")
    )

    mp <- paste0(dp, "mid_block.")
    ap <- paste0(mp, "attentions.0.")
    w$mid <- list(
        resnet1 = resnet(paste0(mp, "resnets.0.")),
        resnet2 = resnet(paste0(mp, "resnets.1.")),
        attn = list(
            gn_w = raw(paste0(ap, "group_norm.weight")),
            gn_b = raw(paste0(ap, "group_norm.bias")),
            q_w = lin(paste0(ap, "to_q.weight")), q_b = raw(paste0(ap, "to_q.bias")),
            k_w = lin(paste0(ap, "to_k.weight")), k_b = raw(paste0(ap, "to_k.bias")),
            v_w = lin(paste0(ap, "to_v.weight")), v_b = raw(paste0(ap, "to_v.bias")),
            out_w = lin(paste0(ap, "to_out.0.weight")),
            out_b = raw(paste0(ap, "to_out.0.bias"))
        )
    )

    w$up_blocks <- lapply(0:3, function(i) {
        bp <- sprintf("%sup_blocks.%d.", dp, i)
        blk <- list(resnets = lapply(0:2, function(j)
            resnet(sprintf("%sresnets.%d.", bp, j))))
        if (has(paste0(bp, "upsamplers.0.conv.weight"))) {
            blk$up_conv_w <- raw(paste0(bp, "upsamplers.0.conv.weight"))
            blk$up_conv_b <- raw(paste0(bp, "upsamplers.0.conv.bias"))
        }
        blk
    })

    w
}
