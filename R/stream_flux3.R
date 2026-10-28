# FLUX 3 on small cards: weights that do not fit stay in (pinned) host
# memory and visit the GPU only while they compute. The Qwen3-VL text
# model streams layer by layer through its encode; the fp8 transformer
# keeps a budget of linears resident and streams the rest inside each
# forward (ltx23_fp8_linear copies a host weight to the input's device).

# How much to stream for a card of `vram_gb` (total). The Qwen3-VL
# encode peaks ~9.2 GB whole, so it streams below 10 GB. The fp8
# transformer (~4.8 GB of fp8 linears) stays whole from 12 GB up; below
# that it keeps vram_gb - 6.5 GB resident, leaving room for the bf16
# residents and a 1 MP edit's activations (8,192 content tokens).
# Measured on a simulated 8 GB card: 1.5 GB resident runs 1024x1024
# text-to-image and the 1 MP edit; 3.5 GB runs text-to-image only.
.flux3_stream_plan <- function(vram_gb) {
    if (!isTRUE(vram_gb > 0)) {
        return(list(stream_text = FALSE, resident_gb = Inf))
    }
    list(stream_text = vram_gb < 10,
         resident_gb = if (vram_gb >= 12) Inf else max(0, vram_gb - 6.5))
}

# Staging pairs for a list of tensors (see staging.R): pinned host copies
# when `pin`, otherwise the tensors' own host storage
.stage_pairs <- function(tensors, pin) {
    pin <- isTRUE(pin) && torch::cuda_is_available()
    lapply(tensors, function(p) {
        if (pin) {
            pinned <- .pin_host(p)
            p$set_data(pinned)
        } else {
            pinned <- p$detach()
        }
        list(live = p, pinned = pinned)
    })
}

# Stream a Qwen3 encoder's layers: one staging set per layer goes on
# the module (its forward swaps each layer in and out); the returned
# pairs (embeddings and final norm) onload for the encode phase
.flux3_text_stream <- function(encoder, pin) {
    layers <- encoder$model$layers
    encoder$layer_stream <- lapply(seq_along(layers), function(i) {
        .stage_pairs(c(layers[[i]]$parameters, layers[[i]]$buffers), pin)
    })
    .stage_pairs(c(encoder$model$embed_tokens$parameters,
                   encoder$model$norm$parameters), pin)
}

# Split a transformer's fp8 linears into a resident set (in module
# order, up to `resident_gb` of fp8 bytes) and a streamed set. Streamed
# weights are pinned in place so their per-forward copies run at DMA
# rate.
.flux3_fp8_split <- function(transformer, resident_gb, pin) {
    mods <- list()
    walk <- function(m) {
        for (child in m$children) {
            if (!is.null(child$weight_fp8)) {
                mods[[length(mods) + 1L]] <<- child
            } else {
                walk(child)
            }
        }
    }
    walk(transformer)
    bytes <- vapply(mods, function(m) m$weight_fp8$numel(), 0)
    keep <- cumsum(bytes) <= resident_gb * 1024 ^ 3
    streamed <- mods[!keep]
    if (isTRUE(pin) && torch::cuda_is_available()) {
        for (m in streamed) {
            m$weight_fp8$set_data(.pin_host(m$weight_fp8))
        }
    }
    list(resident = mods[keep], streamed = streamed,
         resident_gb = sum(bytes[keep]) / 1024 ^ 3,
         streamed_gb = sum(bytes[!keep]) / 1024 ^ 3)
}

# The resident fp8 fields (weights and scales), for staging
.flux3_fp8_fields <- function(mods) {
    unlist(lapply(mods, function(m) list(m$weight_fp8, m$weight_scale)),
           recursive = FALSE)
}

# Move the resident fp8 linears' weights (unstaged pipelines)
.flux3_fp8_to_device <- function(mods, device) {
    for (m in mods) {
        m$weight_fp8 <- m$weight_fp8$to(device = device)
        m$weight_scale <- m$weight_scale$to(device = device)
    }
    invisible(NULL)
}
