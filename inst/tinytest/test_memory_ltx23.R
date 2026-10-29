# ltx23_tune_gc option semantics: defaults land when unset, user-set
# values win, and the reserved rate clamps to [0.20, 0.92]. The cpp
# gate push itself is CUDA-only and exercised implicitly wherever a
# GPU is present.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}

library(diffuseR)

gate_opts <- c("torch.threshold_call_gc",
               "torch.cuda_allocator_allocated_rate",
               "torch.cuda_allocator_allocated_reserved_rate",
               "torch.cuda_allocator_reserved_rate")
old <- options()[gate_opts]
names(old) <- gate_opts
clear <- function() {
  for (o in gate_opts) {
    opt <- list(NULL)
    names(opt) <- o
    options(opt)
  }
}

clear()
rate <- ltx23_tune_gc(footprint_gb = 12, total_gb = 16)
expect_equal(rate, 0.75)
expect_equal(getOption("torch.cuda_allocator_reserved_rate"), 0.75)
expect_equal(getOption("torch.threshold_call_gc"), 16000)
expect_equal(getOption("torch.cuda_allocator_allocated_rate"), 0.95)
expect_equal(getOption("torch.cuda_allocator_allocated_reserved_rate"), 0.95)

# User-set values win; the reserved rate is then not recomputed
clear()
options(torch.cuda_allocator_reserved_rate = 0.5,
        torch.threshold_call_gc = 4000)
rate <- ltx23_tune_gc(footprint_gb = 12, total_gb = 16)
expect_null(rate)
expect_equal(getOption("torch.cuda_allocator_reserved_rate"), 0.5)
expect_equal(getOption("torch.threshold_call_gc"), 4000)

# Clamps: floor 0.20 for small footprints, ceiling 0.92 for tight fits
clear()
expect_equal(ltx23_tune_gc(footprint_gb = 12, total_gb = 100), 0.20)
clear()
expect_equal(ltx23_tune_gc(footprint_gb = 12, total_gb = 12), 0.92)

options(old)

# Profiles: small cards stream NF4 transformer blocks (vram - 4 GB
# resident) and, below 12 GB, the Gemma 4 layers
p16 <- ltx23_memory_profile(16)
expect_equal(p16$name, "high")
expect_equal(p16$resident_gb, Inf)
expect_false(p16$stream_text)
p12 <- ltx23_memory_profile(12)
expect_equal(p12$name, "medium")
expect_equal(p12$precision, "nf4")
expect_equal(p12$resident_gb, 8)
expect_false(p12$stream_text)
p8 <- ltx23_memory_profile(8)
expect_equal(p8$name, "low")
expect_equal(p8$precision, "nf4")
expect_equal(p8$resident_gb, 4)
expect_true(p8$stream_text)
expect_true(ltx23_memory_profile(11)$stream_text)
expect_equal(ltx23_memory_profile(4)$name, "cpu_only")

# Host RAM: plenty keeps everything in one process with host copies;
# less splits the text phase off, then drops the resident host copy
big <- ltx23_memory_profile(8, host_ram_gb = 64)
expect_false(big$split_text)
expect_equal(big$resident_from, "host")
mid <- ltx23_memory_profile(8, host_ram_gb = 24)
expect_true(mid$split_text)
expect_equal(mid$resident_from, "host")
small <- ltx23_memory_profile(12, host_ram_gb = 16)
expect_true(small$split_text)
expect_equal(small$resident_from, "disk")
# Below the render pipeline's 16 GB with the resident part on disk, no
# weights stay in host memory at all
expect_equal(small$stage_from, "disk")
expect_equal(mid$stage_from, "host")
expect_equal(big$stage_from, "host")
expect_equal(ltx23_memory_profile(8, host_ram_gb = 10)$stage_from, "disk")
# Unknown RAM keeps the defaults; the CPU tier never stages
unknown <- ltx23_memory_profile(8, host_ram_gb = NA)
expect_false(unknown$split_text)
expect_equal(unknown$resident_from, "host")
expect_equal(unknown$stage_from, "host")
cpu <- ltx23_memory_profile(4, host_ram_gb = 8)
expect_false(cpu$split_text)
expect_equal(cpu$resident_from, "host")
expect_equal(cpu$stage_from, "host")

# The loader refuses block streaming it cannot do
expect_error(ltx23_load_pipeline("unused", device = "cpu", resident_gb = 4),
             "phase_offload")
expect_error(ltx23_load_pipeline("unused", device = "cpu",
                                 resident_from = "disk"), "phase_offload")
expect_error(ltx23_load_pipeline("unused", device = "cpu",
                                 stage_from = "disk"), "phase_offload")
expect_error(load_gemma4_nf4("unused", device = "cuda", stage_from = "disk"),
             "device = \"cpu\"")

# The split text phase needs connectors, and an encoder or raw states
no_conn <- structure(list(), class = "ltx23_pipeline")
expect_error(ltx23_text_embeds("p", no_conn), "connectors")
fake <- structure(list(connectors = list()), class = "ltx23_pipeline")
expect_error(ltx23_text_embeds("p", fake), "prompt_embeds")
