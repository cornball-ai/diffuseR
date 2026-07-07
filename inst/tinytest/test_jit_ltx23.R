# Parity of the TorchScript block stack (R/jit_ltx23.R) against the
# eager NF4 transformer block. The compiled path must reproduce the
# eager path bit-for-bit up to SDPA-vs-materialized-attention rounding.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}

library(diffuseR)
torch::torch_manual_seed(7)

# The prompt KV modulation tables are dim-sized and apply directly to
# the text streams, so cross_attention_dim == dim (as in the real
# model: 4096/4096 video, 2048/2048 audio)
dim <- 32L; heads <- 2L; head_dim <- 16L; cross_dim <- 32L
adim <- 16L; aheads <- 2L; ahead_dim <- 8L; across_dim <- 16L
B <- 2L; Sv <- 6L; Sa <- 5L; St <- 4L

make_block <- function() {
  blk <- ltx23_transformer_block(
    dim = dim, num_attention_heads = heads, attention_head_dim = head_dim,
    cross_attention_dim = cross_dim,
    audio_dim = adim, audio_num_attention_heads = aheads,
    audio_attention_head_dim = ahead_dim,
    audio_cross_attention_dim = across_dim
  )
  blk$eval()

  # Swap every cast linear for an NF4 module exactly like the loader does
  swap_lin <- function(parent, leaf) {
    old <- diffuseR:::.ltx23_walk_module(parent, leaf)
    q <- ltx23_nf4_quantize(old$weight)
    m <- ltx23_nf4_linear(old$weight$shape[1], old$weight$shape[2],
      bias = !is.null(old$bias))
    if (!is.null(old$bias)) m$bias <- old$bias
    m$set_nf4_weight(q$packed, q$absmax)
    do.call(`$<-`, list(parent, leaf, m))
  }
  for (attn in list(blk$attn1, blk$audio_attn1, blk$attn2, blk$audio_attn2,
                    blk$audio_to_video_attn, blk$video_to_audio_attn)) {
    swap_lin(attn, "to_q")
    swap_lin(attn, "to_k")
    swap_lin(attn, "to_v")
    swap_lin(attn$to_out, "0")
  }
  for (ff in list(blk$ff, blk$audio_ff)) {
    swap_lin(ff$net[[1]], "proj")
    swap_lin(ff$net, "2")
  }
  blk
}

blk1 <- make_block()
blk2 <- make_block()
expect_true(diffuseR:::.ltx23_jit_block_ok(blk1))

# The model's split-rope embedder emits per-head 4D freqs [B, H, T, r]
rope_pair <- function(s, n_heads, r) {
  ang <- torch::torch_rand(B, n_heads, s, r) * 6.28
  list(torch::torch_cos(ang), torch::torch_sin(ang))
}

h <- torch::torch_randn(B, Sv, dim)
ah <- torch::torch_randn(B, Sa, adim)
enc <- torch::torch_randn(B, St, cross_dim)
aenc <- torch::torch_randn(B, St, across_dim)
temb <- torch::torch_randn(B, 1L, 9L * dim)
temb_a <- torch::torch_randn(B, 1L, 9L * adim)
tcss <- torch::torch_randn(B, 1L, 4L * dim)
tcass <- torch::torch_randn(B, 1L, 4L * adim)
tcg <- torch::torch_randn(B, 1L, dim)
tcag <- torch::torch_randn(B, 1L, adim)
tp <- torch::torch_randn(B, 1L, 2L * dim)
tpa <- torch::torch_randn(B, 1L, 2L * adim)
v_rope <- rope_pair(Sv, heads, head_dim %/% 2L)
a_rope <- rope_pair(Sa, aheads, ahead_dim %/% 2L)
cav_rope <- rope_pair(Sv, aheads, ahead_dim %/% 2L)
caa_rope <- rope_pair(Sa, aheads, ahead_dim %/% 2L)
# Additive [B, 1, S] mask with one text token masked out
enc_mask <- torch::torch_zeros(B, 1L, St)
enc_mask[, , St] <- -10000
aenc_mask <- torch::torch_zeros(B, 1L, St)

eager <- function(blk, h, ah) {
  blk(
    hidden_states = h, audio_hidden_states = ah,
    encoder_hidden_states = enc, audio_encoder_hidden_states = aenc,
    temb = temb, temb_audio = temb_a,
    temb_ca_scale_shift = tcss, temb_ca_audio_scale_shift = tcass,
    temb_ca_gate = tcg, temb_ca_audio_gate = tcag,
    temb_prompt = tp, temb_prompt_audio = tpa,
    video_rotary_emb = v_rope, audio_rotary_emb = a_rope,
    ca_video_rotary_emb = cav_rope, ca_audio_rotary_emb = caa_rope,
    encoder_attention_mask = enc_mask,
    audio_encoder_attention_mask = aenc_mask
  )
}

jit <- function(blocks, h, ah) {
  diffuseR:::.ltx23_jit_run_stack(
    blocks, h, ah, enc, aenc,
    temb, temb_a, tcss, tcass, tcg, tcag, tp, tpa,
    v_rope, a_rope, cav_rope, caa_rope,
    encoder_attention_mask = enc_mask,
    audio_encoder_attention_mask = aenc_mask
  )
}

max_abs_diff <- function(a, b) as.numeric((a - b)$abs()$max())

torch::with_no_grad({
  ref1 <- eager(blk1, h, ah)
  out1 <- jit(list(blk1), h, ah)
})
expect_equal(as.integer(out1[[1]]$shape), as.integer(ref1[[1]]$shape))
expect_equal(as.integer(out1[[2]]$shape), as.integer(ref1[[2]]$shape))
expect_true(max_abs_diff(out1[[1]], ref1[[1]]) < 1e-4)
expect_true(max_abs_diff(out1[[2]], ref1[[2]]) < 1e-4)

# Two-block stack must equal two sequential eager blocks (checks the
# per-block base-offset arithmetic in the weight list)
torch::with_no_grad({
  mid <- eager(blk1, h, ah)
  ref2 <- eager(blk2, mid[[1]], mid[[2]])
  out2 <- jit(list(blk1, blk2), h, ah)
})
expect_true(max_abs_diff(out2[[1]], ref2[[1]]) < 1e-4)
expect_true(max_abs_diff(out2[[2]], ref2[[2]]) < 1e-4)

# The mask must actually mask: moving the -10000 changes the output
enc_mask2 <- torch::torch_zeros(B, 1L, St)
enc_mask2[, , 1L] <- -10000
torch::with_no_grad({
  out_m <- diffuseR:::.ltx23_jit_run_stack(
    list(blk1), h, ah, enc, aenc,
    temb, temb_a, tcss, tcass, tcg, tcag, tp, tpa,
    v_rope, a_rope, cav_rope, caa_rope,
    encoder_attention_mask = enc_mask2,
    audio_encoder_attention_mask = aenc_mask
  )
})
expect_true(max_abs_diff(out_m[[1]], out1[[1]]) > 1e-6)

# NULL masks accepted
torch::with_no_grad({
  ref_nm <- blk1(
    hidden_states = h, audio_hidden_states = ah,
    encoder_hidden_states = enc, audio_encoder_hidden_states = aenc,
    temb = temb, temb_audio = temb_a,
    temb_ca_scale_shift = tcss, temb_ca_audio_scale_shift = tcass,
    temb_ca_gate = tcg, temb_ca_audio_gate = tcag,
    temb_prompt = tp, temb_prompt_audio = tpa,
    video_rotary_emb = v_rope, audio_rotary_emb = a_rope,
    ca_video_rotary_emb = cav_rope, ca_audio_rotary_emb = caa_rope
  )
  out_nm <- diffuseR:::.ltx23_jit_run_stack(
    list(blk1), h, ah, enc, aenc,
    temb, temb_a, tcss, tcass, tcg, tcag, tp, tpa,
    v_rope, a_rope, cav_rope, caa_rope
  )
})
expect_true(max_abs_diff(out_nm[[1]], ref_nm[[1]]) < 1e-4)
expect_true(max_abs_diff(out_nm[[2]], ref_nm[[2]]) < 1e-4)

# 3D whole-vector rope layout (the apply fn's other branch) also matches
rope3 <- function(s, r) {
  ang <- torch::torch_rand(1L, s, r) * 6.28
  list(torch::torch_cos(ang), torch::torch_sin(ang))
}
v3 <- rope3(Sv, (heads * head_dim) %/% 2L)
a3 <- rope3(Sa, (aheads * ahead_dim) %/% 2L)
cav3 <- rope3(Sv, (aheads * ahead_dim) %/% 2L)
caa3 <- rope3(Sa, (aheads * ahead_dim) %/% 2L)
torch::with_no_grad({
  ref3 <- blk1(
    hidden_states = h, audio_hidden_states = ah,
    encoder_hidden_states = enc, audio_encoder_hidden_states = aenc,
    temb = temb, temb_audio = temb_a,
    temb_ca_scale_shift = tcss, temb_ca_audio_scale_shift = tcass,
    temb_ca_gate = tcg, temb_ca_audio_gate = tcag,
    temb_prompt = tp, temb_prompt_audio = tpa,
    video_rotary_emb = v3, audio_rotary_emb = a3,
    ca_video_rotary_emb = cav3, ca_audio_rotary_emb = caa3
  )
  out3 <- diffuseR:::.ltx23_jit_run_stack(
    list(blk1), h, ah, enc, aenc,
    temb, temb_a, tcss, tcass, tcg, tcag, tp, tpa,
    v3, a3, cav3, caa3
  )
})
expect_true(max_abs_diff(out3[[1]], ref3[[1]]) < 1e-4)
expect_true(max_abs_diff(out3[[2]], ref3[[2]]) < 1e-4)

# Per-token temb (i2v conditioning: per-token video timestep) matches
temb_tok <- torch::torch_randn(B, Sv, 9L * dim)
torch::with_no_grad({
  ref_pt <- blk1(
    hidden_states = h, audio_hidden_states = ah,
    encoder_hidden_states = enc, audio_encoder_hidden_states = aenc,
    temb = temb_tok, temb_audio = temb_a,
    temb_ca_scale_shift = tcss, temb_ca_audio_scale_shift = tcass,
    temb_ca_gate = tcg, temb_ca_audio_gate = tcag,
    temb_prompt = tp, temb_prompt_audio = tpa,
    video_rotary_emb = v_rope, audio_rotary_emb = a_rope,
    ca_video_rotary_emb = cav_rope, ca_audio_rotary_emb = caa_rope
  )
  out_pt <- diffuseR:::.ltx23_jit_run_stack(
    list(blk1), h, ah, enc, aenc,
    temb_tok, temb_a, tcss, tcass, tcg, tcag, tp, tpa,
    v_rope, a_rope, cav_rope, caa_rope
  )
})
expect_true(max_abs_diff(out_pt[[1]], ref_pt[[1]]) < 1e-4)
expect_true(max_abs_diff(out_pt[[2]], ref_pt[[2]]) < 1e-4)

# Compact conditioned temb (2 variants + per-token index) must equal
# the eager block run with the explicitly expanded per-token temb
temb2 <- torch::torch_randn(B, 2L, 9L * dim)
cond_idx <- torch::torch_tensor(c(1L, 1L, 0L, 0L, 0L, 0L),
  dtype = torch::torch_long()) # first two tokens conditioned
temb_full <- temb2$index_select(2L, cond_idx$add(1L))
torch::with_no_grad({
  ref_ci <- blk1(
    hidden_states = h, audio_hidden_states = ah,
    encoder_hidden_states = enc, audio_encoder_hidden_states = aenc,
    temb = temb_full, temb_audio = temb_a,
    temb_ca_scale_shift = tcss, temb_ca_audio_scale_shift = tcass,
    temb_ca_gate = tcg, temb_ca_audio_gate = tcag,
    temb_prompt = tp, temb_prompt_audio = tpa,
    video_rotary_emb = v_rope, audio_rotary_emb = a_rope,
    ca_video_rotary_emb = cav_rope, ca_audio_rotary_emb = caa_rope
  )
  out_ci <- diffuseR:::.ltx23_jit_run_stack(
    list(blk1), h, ah, enc, aenc,
    temb2, temb_a, tcss, tcass, tcg, tcag, tp, tpa,
    v_rope, a_rope, cav_rope, caa_rope,
    cond_token_index = cond_idx
  )
})
expect_true(max_abs_diff(out_ci[[1]], ref_ci[[1]]) < 1e-4)
expect_true(max_abs_diff(out_ci[[2]], ref_ci[[2]]) < 1e-4)
