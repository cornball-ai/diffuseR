# LTX on small cards: the transformer keeps a budget of its leading
# blocks on the GPU and streams the rest from (pinned) host memory, one
# block per call, every step. Blocks are ~0.25 GB in NF4 (22B over 48
# blocks), so streaming k of them costs k * 0.25 GB of DMA per step.

# Run f() with a streamed unit of `module` (module$unit_stream[[unit]],
# a staging set) on `device`; a plain call when the unit is not streamed
.unit_call <- function(module, unit, device, f) {
    st <- module$unit_stream[[unit]]
    if (is.null(st)) {
        return(f())
    }
    .staged_onload(st, device)
    on.exit({
        .staged_offload(st)
        gc(verbose = FALSE, full = FALSE)
    })
    f()
}

# Stream the text connectors (5.9 GB in bf16): the two input projections
# (1.5 and 0.8 GB) as units, each connector's blocks one at a time.
# Returns the staging pairs for the rest (registers, norms, rope).
.ltx23_connector_stream <- function(connectors, pin) {
    units <- c("video_text_proj_in", "audio_text_proj_in")
    connectors$unit_stream <- stats::setNames(lapply(units, function(u) {
        m <- connectors[[u]]
        .stage_pairs(c(m$parameters, m$buffers), pin)
    }), units)
    for (cn in c("video_connector", "audio_connector")) {
        con <- connectors[[cn]]
        con$layer_stream <- lapply(seq_along(con$transformer_blocks), function(i) {
            b <- con$transformer_blocks[[i]]
            .stage_pairs(c(b$parameters, b$buffers), pin)
        })
    }
    tensors <- c(connectors$parameters, connectors$buffers)
    streamed <- startsWith(names(tensors), "video_text_proj_in.") |
    startsWith(names(tensors), "audio_text_proj_in.") |
    startsWith(names(tensors), "video_connector.transformer_blocks.") |
    startsWith(names(tensors), "audio_connector.transformer_blocks.")
    .stage_pairs(tensors[!streamed], pin)
}

# Bytes held by a module's parameters and buffers
.module_bytes <- function(module) {
    ts <- c(module$parameters, module$buffers)
    sum(vapply(ts, function(t) t$numel() * t$element_size(), 0))
}

# Split `transformer` so at most `resident_gb` of it (non-block tensors
# plus its leading blocks) stays with the phase onload. Sets
# transformer$block_stream (per-block staging for the streamed blocks,
# consumed by the forward) and returns the staging pairs for everything
# else, which replace the transformer's whole-module staging set.
.ltx23_block_stream <- function(transformer, resident_gb, pin) {
    blocks <- transformer$transformer_blocks
    n <- length(blocks)
    block_bytes <- vapply(seq_len(n), function(i) .module_bytes(blocks[[i]]), 0)
    other_bytes <- .module_bytes(transformer) - sum(block_bytes)
    budget <- resident_gb * 1024 ^ 3 - other_bytes
    n_resident <- sum(cumsum(block_bytes) <= budget)

    streamed <- if (n_resident < n) seq(n_resident + 1L, n) else integer(0)
    stream_staging <- lapply(streamed, function(i) {
        .stage_pairs(c(blocks[[i]]$parameters, blocks[[i]]$buffers), pin)
    })
    # Everything not in a streamed block, by identity of the tensor names
    names_all <- c(names(transformer$parameters), names(transformer$buffers))
    tensors <- c(transformer$parameters, transformer$buffers)
    prefix <- paste0("transformer_blocks.", streamed - 1L, ".")
    in_stream <- Reduce(`|`,
                        lapply(prefix, function(p) startsWith(names_all, p)),
                        FALSE)
    transformer$block_stream <- if (length(streamed)) {
        list(n_resident = n_resident, staging = stream_staging)
    } else {
        NULL
    }
    list(staging = .stage_pairs(tensors[!in_stream], pin),
         n_resident = n_resident, n_streamed = length(streamed),
         resident_gb = (other_bytes + sum(block_bytes[seq_len(n_resident)])) / 1024 ^ 3,
         streamed_gb = sum(block_bytes[streamed]) / 1024 ^ 3)
}
