# .ltx23_text_encode_device: which device the prompt encode runs on for
# the text_encoder txt2vid_ltx2 was handed. Pure decision, no torch, no
# GPU, no real encode -- the point is to pin the rule that a resident
# (preloaded, pinned) encoder stages to the card instead of encoding on
# CPU, which is what the gpuhost/resident path silently did before.

library(diffuseR)
f <- diffuseR:::.ltx23_text_encode_device

# A PATH loads fresh onto whatever device was asked for.
expect_equal(f("/models/gemma3-nf4", "cuda"), "cuda")
expect_equal(f("/models/gemma3-nf4", "cuda:0"), "cuda:0")
expect_equal(f("/models/gemma3-nf4", "cpu"), "cpu")

# A PRELOADED object is only encoded on the card when it carries the
# pinned staging set (the resident loader's pin = TRUE). This is the case
# that regressed: the resident/gpuhost encoder HAS staging and a cuda
# request, so it must stage to the card, not fall to CPU.
staged <- structure(list(), staging = list(TRUE))
expect_equal(f(staged, "cuda"), "cuda")
expect_equal(f(staged, "cuda:0"), "cuda:0")
# An explicit cpu request is still honoured even with staging present.
expect_equal(f(staged, "cpu"), "cpu")

# A preloaded object with NO staging cannot run on the card -- its weights
# sit on the host and a cuda request would be a device mismatch -- so it
# degrades to CPU. This is the safety the blunt `else "cpu"` was protecting,
# preserved for exactly this case.
bare <- structure(list())
expect_equal(f(bare, "cuda"), "cpu")
expect_equal(f(bare, "cpu"), "cpu")
