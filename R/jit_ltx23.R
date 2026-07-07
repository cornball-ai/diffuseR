#' LTX-2.3 JIT Block Stack
#'
#' TorchScript compilation of the 48-block NF4 transformer step (cf.
#' the torch skill's JIT-decode pattern proven in whisper and
#' chatterbox). Eager execution crosses R -> lantern per op (~190 us
#' each) and leaves every intermediate as an R tensor handle that only
#' dies at gc(); at high resolution that forces a per-block gc()
#' costing the vast majority of step time. Compiled, the whole block
#' stack is one crossing: intermediates are freed eagerly by libtorch,
#' no R garbage accumulates, no per-block gc is needed, and attention
#' runs through the fused \code{scaled_dot_product_attention} kernel
#' instead of a materialized score matrix.
#'
#' Weights are passed per call as a flat \code{List[Tensor]} (borrowed
#' by reference, no copies) with a fixed per-block layout; the packer
#' and the TorchScript indices must stay in lockstep (parity-tested).
#'
#' @name jit_ltx23
NULL

# Per-attention tensor layout (16 slots):
#   0 gate_w, 1 gate_b,
#   2 q_packed, 3 q_absmax, 4 q_bias,
#   5 k_packed, 6 k_absmax, 7 k_bias,
#   8 v_packed, 9 v_absmax, 10 v_bias,
#   11 out_packed, 12 out_absmax, 13 out_bias,
#   14 norm_q_w, 15 norm_k_w
# Per-block layout (114 slots): attn1 @0, audio_attn1 @16, attn2 @32,
# audio_attn2 @48, audio_to_video_attn @64, video_to_audio_attn @80,
# ff @96 (proj_packed, proj_absmax, proj_bias, net2_packed,
# net2_absmax, net2_bias), audio_ff @102, then the modulation tables:
# 108 scale_shift_table, 109 audio_scale_shift_table,
# 110 prompt_scale_shift_table, 111 audio_prompt_scale_shift_table,
# 112 video_a2v_cross_attn_scale_shift_table,
# 113 audio_a2v_cross_attn_scale_shift_table.
.ltx23_jit_slots_per_block <- 114L

.ltx23_jit_source <- function() {
    "
def nf4_lin(x: Tensor, packed: Tensor, absmax: Tensor, bias: Tensor, table: Tensor) -> Tensor:
    rows = bias.size(0)
    n = packed.size(0)
    step = 4194304
    outs: List[Tensor] = []
    i = 0
    while i < n:
        j = min(i + step, n)
        chunk = packed.narrow(0, i, j - i)
        hi = torch.bitwise_right_shift(chunk, 4).long()
        lo = torch.bitwise_and(chunk, 15).long()
        idx = torch.stack([hi, lo], -1).flatten()
        vals = torch.index_select(table, 0, idx)
        sc = absmax.narrow(0, i * 2 // 64, (j - i) * 2 // 64)
        outs.append((vals.reshape(-1, 64) * sc.unsqueeze(1)).flatten().type_as(x))
        i = j
    w = torch.cat(outs, 0).reshape([rows, n * 2 // rows])
    return torch.linear(x, w, bias)

def rmsn(x: Tensor) -> Tensor:
    v = x.float().pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(v + 1e-6)).type_as(x)

def rmsn_w(x: Tensor, w: Tensor) -> Tensor:
    v = x.float().pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(v + 1e-6)).type_as(w) * w

def rope(x: Tensor, cs: Tensor, sn: Tensor) -> Tensor:
    if cs.dim() == 4:
        # Per-head split layout [B, H, T, r]: x [B, T, H*D] -> [B, H, T, D]
        b = cs.size(0)
        hh = cs.size(1)
        t = cs.size(2)
        xh = x.reshape([b, t, hh, -1]).transpose(1, 2)
        r = xh.size(-1) // 2
        xf = xh.narrow(-1, 0, r).float()
        xs = xh.narrow(-1, r, r).float()
        o = torch.cat([xf * cs - xs * sn, xs * cs + xf * sn], -1)
        return o.transpose(1, 2).reshape([b, t, -1]).type_as(x)
    r2 = x.size(-1) // 2
    xf2 = x.narrow(-1, 0, r2).float()
    xs2 = x.narrow(-1, r2, r2).float()
    return torch.cat([xf2 * cs - xs2 * sn, xs2 * cs + xf2 * sn], -1).type_as(x)

def mods(tbl: Tensor, temb: Tensor, num: int) -> Tensor:
    b = temb.size(0)
    t = temb.size(1)
    return tbl.unsqueeze(0).unsqueeze(0) + temb.reshape([b, t, num, -1])

def attn_nf4(x: Tensor, ctx: Tensor, ws: List[Tensor], base: int, table: Tensor, heads: int,
             q_cos: Optional[Tensor], q_sin: Optional[Tensor],
             k_cos: Optional[Tensor], k_sin: Optional[Tensor],
             mask: Optional[Tensor]) -> Tensor:
    gl = torch.linear(x, ws[base], ws[base + 1])
    q = rmsn_w(nf4_lin(x, ws[base + 2], ws[base + 3], ws[base + 4], table), ws[base + 14])
    k = rmsn_w(nf4_lin(ctx, ws[base + 5], ws[base + 6], ws[base + 7], table), ws[base + 15])
    v = nf4_lin(ctx, ws[base + 8], ws[base + 9], ws[base + 10], table)
    if q_cos is not None and q_sin is not None:
        q = rope(q, q_cos, q_sin)
    if k_cos is not None and k_sin is not None:
        k = rope(k, k_cos, k_sin)
    qh = q.unflatten(-1, [heads, -1]).transpose(1, 2)
    kh = k.unflatten(-1, [heads, -1]).transpose(1, 2)
    vh = v.unflatten(-1, [heads, -1]).transpose(1, 2)
    o = torch.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)
    o = o.transpose(1, 2).flatten(2).type_as(x)
    gates = torch.sigmoid(gl) * 2.0
    o = (o.unflatten(-1, [heads, -1]) * gates.unsqueeze(-1)).flatten(2)
    return nf4_lin(o, ws[base + 11], ws[base + 12], ws[base + 13], table)

def ff_nf4(x: Tensor, ws: List[Tensor], base: int, table: Tensor) -> Tensor:
    h = torch.gelu(nf4_lin(x, ws[base], ws[base + 1], ws[base + 2], table), approximate=\"tanh\")
    return nf4_lin(h, ws[base + 3], ws[base + 4], ws[base + 5], table)

def block_nf4(h: Tensor, ah: Tensor, enc: Tensor, aenc: Tensor,
              temb: Tensor, temb_a: Tensor,
              tcss: Tensor, tcass: Tensor, tcg: Tensor, tcag: Tensor,
              tp: Tensor, tpa: Tensor,
              v_cos: Tensor, v_sin: Tensor, a_cos: Tensor, a_sin: Tensor,
              cav_cos: Tensor, cav_sin: Tensor, caa_cos: Tensor, caa_sin: Tensor,
              enc_mask: Optional[Tensor], aenc_mask: Optional[Tensor],
              ws: List[Tensor], base: int, table: Tensor,
              heads: int, aheads: int) -> Tuple[Tensor, Tensor]:
    vada = mods(ws[base + 108], temb, 9)
    aada = mods(ws[base + 109], temb_a, 9)

    nh = rmsn(h) * (vada.select(2, 1) + 1.0) + vada.select(2, 0)
    ax = attn_nf4(nh, nh, ws, base, table, heads, v_cos, v_sin, v_cos, v_sin, None)
    h = h + ax * vada.select(2, 2)

    nah = rmsn(ah) * (aada.select(2, 1) + 1.0) + aada.select(2, 0)
    aax = attn_nf4(nah, nah, ws, base + 16, table, aheads, a_cos, a_sin, a_cos, a_sin, None)
    ah = ah + aax * aada.select(2, 2)

    pada = mods(ws[base + 110], tp, 2)
    apada = mods(ws[base + 111], tpa, 2)

    nh = rmsn(h) * (vada.select(2, 7) + 1.0) + vada.select(2, 6)
    encm = enc * (pada.select(2, 1) + 1.0) + pada.select(2, 0)
    ax = attn_nf4(nh, encm, ws, base + 32, table, heads, None, None, None, None, enc_mask)
    h = h + ax * vada.select(2, 8)

    nah = rmsn(ah) * (aada.select(2, 7) + 1.0) + aada.select(2, 6)
    aencm = aenc * (apada.select(2, 1) + 1.0) + apada.select(2, 0)
    aax = attn_nf4(nah, aencm, ws, base + 48, table, aheads, None, None, None, None, aenc_mask)
    ah = ah + aax * aada.select(2, 8)

    nh = rmsn(h)
    nah = rmsn(ah)
    vca = mods(ws[base + 112].narrow(0, 0, 4), tcss, 4)
    vcg = mods(ws[base + 112].narrow(0, 4, 1), tcg, 1)
    aca = mods(ws[base + 113].narrow(0, 0, 4), tcass, 4)
    acg = mods(ws[base + 113].narrow(0, 4, 1), tcag, 1)

    mnh = nh * (vca.select(2, 0) + 1.0) + vca.select(2, 1)
    mna = nah * (aca.select(2, 0) + 1.0) + aca.select(2, 1)
    a2v = attn_nf4(mnh, mna, ws, base + 64, table, aheads, cav_cos, cav_sin, caa_cos, caa_sin, None)
    h = h + vcg.select(2, 0) * a2v

    mnh = nh * (vca.select(2, 2) + 1.0) + vca.select(2, 3)
    mna = nah * (aca.select(2, 2) + 1.0) + aca.select(2, 3)
    v2a = attn_nf4(mna, mnh, ws, base + 80, table, aheads, caa_cos, caa_sin, cav_cos, cav_sin, None)
    ah = ah + acg.select(2, 0) * v2a

    nh = rmsn(h) * (vada.select(2, 4) + 1.0) + vada.select(2, 3)
    h = h + ff_nf4(nh, ws, base + 96, table) * vada.select(2, 5)

    nah = rmsn(ah) * (aada.select(2, 4) + 1.0) + aada.select(2, 3)
    ah = ah + ff_nf4(nah, ws, base + 102, table) * aada.select(2, 5)

    return (h, ah)

def stack_nf4(h: Tensor, ah: Tensor, enc: Tensor, aenc: Tensor,
              temb: Tensor, temb_a: Tensor,
              tcss: Tensor, tcass: Tensor, tcg: Tensor, tcag: Tensor,
              tp: Tensor, tpa: Tensor,
              v_cos: Tensor, v_sin: Tensor, a_cos: Tensor, a_sin: Tensor,
              cav_cos: Tensor, cav_sin: Tensor, caa_cos: Tensor, caa_sin: Tensor,
              enc_mask: Optional[Tensor], aenc_mask: Optional[Tensor],
              ws: List[Tensor], table: Tensor,
              n_blocks: int, heads: int, aheads: int) -> Tuple[Tensor, Tensor]:
    i = 0
    while i < n_blocks:
        h, ah = block_nf4(h, ah, enc, aenc, temb, temb_a, tcss, tcass, tcg, tcag,
                          tp, tpa, v_cos, v_sin, a_cos, a_sin,
                          cav_cos, cav_sin, caa_cos, caa_sin,
                          enc_mask, aenc_mask, ws, i * 114, table, heads, aheads)
        i += 1
    return (h, ah)
"
}

# Compile once per session
.ltx23_jit_env <- new.env(parent = emptyenv())

.ltx23_jit_unit <- function() {
    unit <- .ltx23_jit_env$unit
    if (is.null(unit)) {
        unit <- torch::jit_compile(.ltx23_jit_source())
        .ltx23_jit_env$unit <- unit
    }
    unit
}

# Flat List[Tensor] for one attention module (16 slots)
.ltx23_jit_pack_attn <- function(attn) {
    list(attn$to_gate_logits$weight, attn$to_gate_logits$bias,
         attn$to_q$weight_nf4, attn$to_q$weight_absmax, attn$to_q$bias,
         attn$to_k$weight_nf4, attn$to_k$weight_absmax, attn$to_k$bias,
         attn$to_v$weight_nf4, attn$to_v$weight_absmax, attn$to_v$bias,
         attn$to_out[[1]]$weight_nf4, attn$to_out[[1]]$weight_absmax,
         attn$to_out[[1]]$bias, attn$norm_q$weight, attn$norm_k$weight)
}

.ltx23_jit_pack_ff <- function(ff) {
    list(ff$net[[1]]$proj$weight_nf4, ff$net[[1]]$proj$weight_absmax,
         ff$net[[1]]$proj$bias, ff$net[[3]]$weight_nf4,
         ff$net[[3]]$weight_absmax, ff$net[[3]]$bias)
}

#' Pack a transformer block's weights for the JIT stack
#'
#' Returns the block's tensors in the fixed 114-slot layout consumed by
#' the compiled \code{stack_nf4}/\code{block_nf4} TorchScript functions.
#' Tensor handles are borrowed, not copied.
#'
#' @param block An NF4-quantized \code{ltx23_transformer_block}.
#'
#' @return List of 114 tensors.
#'
#' @keywords internal
.ltx23_jit_pack_block <- function(block) {
    c(
        .ltx23_jit_pack_attn(block$attn1),
        .ltx23_jit_pack_attn(block$audio_attn1),
        .ltx23_jit_pack_attn(block$attn2),
        .ltx23_jit_pack_attn(block$audio_attn2),
        .ltx23_jit_pack_attn(block$audio_to_video_attn),
        .ltx23_jit_pack_attn(block$video_to_audio_attn),
        .ltx23_jit_pack_ff(block$ff),
        .ltx23_jit_pack_ff(block$audio_ff),
        list(block$scale_shift_table, block$audio_scale_shift_table,
             block$prompt_scale_shift_table,
             block$audio_prompt_scale_shift_table,
             block$video_a2v_cross_attn_scale_shift_table,
             block$audio_a2v_cross_attn_scale_shift_table)
    )
}

# A block is JIT-eligible when every cast linear is NF4 (loader swap
# applied) and the gated/adaln 2.3 features the script bakes in are on
.ltx23_jit_block_ok <- function(block) {
    inherits(block$attn1$to_q, "ltx23_nf4_linear") &&
    inherits(block$ff$net[[3]], "ltx23_nf4_linear") &&
    isTRUE(block$attn1$apply_gated_attention) &&
    isTRUE(block$video_cross_attn_adaln) &&
    isTRUE(block$audio_cross_attn_adaln)
}

# NF4 level table on the right device (cached per device)
.ltx23_jit_table <- function(device) {
    key <- paste(device$type, device$index %||% 0L, sep = "|")
    tbl <- .ltx23_jit_env[[key]]
    if (is.null(tbl)) {
        tbl <- torch::torch_tensor(.ltx23_nf4_table,
                                   dtype = torch::torch_float32(),
                                   device = device)
        .ltx23_jit_env[[key]] <- tbl
    }
    tbl
}

#' Run the block stack through the compiled TorchScript path
#'
#' One R-to-libtorch crossing for all blocks: no per-op dispatch, no R
#' tensor garbage, fused SDPA. Masks must already be additive
#' \code{[B, 1, 1, S]} (or NULL); rope tensors are the \code{[.., r]}
#' cos/sin pairs used by the eager path.
#'
#' @return list(hidden_states, audio_hidden_states)
#'
#' @keywords internal
.ltx23_jit_run_stack <- function(blocks, hidden_states, audio_hidden_states,
                                 encoder_hidden_states,
                                 audio_encoder_hidden_states, temb,
                                 temb_audio, temb_ca_scale_shift,
                                 temb_ca_audio_scale_shift, temb_ca_gate,
                                 temb_ca_audio_gate, temb_prompt,
                                 temb_prompt_audio, video_rotary_emb,
                                 audio_rotary_emb, ca_video_rotary_emb,
                                 ca_audio_rotary_emb,
                                 encoder_attention_mask = NULL,
                                 audio_encoder_attention_mask = NULL) {
    unit <- .ltx23_jit_unit()
    # unname: an nn_module_list yields named children, and a named R
    # list marshals to TorchScript as Dict[str, Tensor], not List[Tensor]
    ws <- unname(do.call(c, lapply(blocks, .ltx23_jit_pack_block)))
    table <- .ltx23_jit_table(hidden_states$device)
    heads <- blocks[[1]]$attn1$heads
    aheads <- blocks[[1]]$audio_attn1$heads

    # Eager attention takes [B, 1, S] additive masks and unsqueezes to
    # [B, 1, 1, S]; SDPA wants them pre-broadcast
    fix_mask <- function(m) {
        if (!is.null(m) && m$ndim == 3L) {
            m$unsqueeze(2L)
        } else {
            m
        }
    }

    res <- unit$stack_nf4(hidden_states, audio_hidden_states,
                          encoder_hidden_states, audio_encoder_hidden_states,
                          temb, temb_audio, temb_ca_scale_shift,
                          temb_ca_audio_scale_shift, temb_ca_gate,
                          temb_ca_audio_gate, temb_prompt, temb_prompt_audio,
                          video_rotary_emb[[1]], video_rotary_emb[[2]],
                          audio_rotary_emb[[1]], audio_rotary_emb[[2]],
                          ca_video_rotary_emb[[1]], ca_video_rotary_emb[[2]],
                          ca_audio_rotary_emb[[1]], ca_audio_rotary_emb[[2]],
                          fix_mask(encoder_attention_mask),
                          fix_mask(audio_encoder_attention_mask), ws, table,
                          torch::jit_scalar(length(blocks)),
                          torch::jit_scalar(as.integer(heads)),
                          torch::jit_scalar(as.integer(aheads)))
    list(res[[1]], res[[2]])
}
