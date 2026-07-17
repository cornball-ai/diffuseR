#' Re-shard a large safetensors file into sub-2 GB shards
#'
#' Splits a single \code{.safetensors} file into diffusers-style shards
#' (\code{<base>-00001-of-000NN.safetensors} plus a
#' \code{<base>.safetensors.index.json} weight map) so it loads on stock
#' CRAN safetensors, which overflows a 32-bit offset on any file at or
#' above 2^31 bytes. Reading the oversize source requires a
#' fork-patched safetensors (a build-machine step); the shards it writes
#' are fork-free to read. Used to host large fp16 diffusers weights
#' (e.g. the 5 GB SDXL UNet) unchanged, without quantization.
#'
#' @param input Path to the source \code{.safetensors} file, or a
#'   directory containing \code{<base>.safetensors}.
#' @param output_dir Output directory for the shards + index.
#' @param base Shard basename (default
#'   \code{"diffusion_pytorch_model"}).
#' @param shard_bytes Target shard size; the default 1.9e9 keeps each
#'   shard under the ~2.15 GB ceiling.
#' @param verbose Logical.
#'
#' @return Invisibly, the path to the written index.json.
#'
#' @export
reshard_safetensors <- function(input, output_dir,
                                base = "diffusion_pytorch_model",
                                shard_bytes = 1.9e9, verbose = TRUE) {
    if (!requireNamespace("safetensors", quietly = TRUE)) {
        stop("The safetensors package is required to reshard.")
    }
    input <- path.expand(input)
    if (dir.exists(input)) {
        input <- file.path(input, paste0(base, ".safetensors"))
    }
    if (!file.exists(input)) {
        stop("No safetensors file at ", input)
    }
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    handle <- safetensors::safetensors$new(input, framework = "torch")
    keys <- setdiff(handle$keys(), "__metadata__")

    shard <- list()
    shard_size <- 0
    shard_idx <- 0L
    key_to_shard <- integer(0) # 1-based shard index per key (named)
    total_size <- 0

    flush <- function() {
        if (!length(shard)) {
            return()
        }
        shard_idx <<- shard_idx + 1L
        # temporary name; renamed once the shard total is known
        fname <- sprintf("%s-%05d.tmp.safetensors", base, shard_idx)
        safetensors::safe_save_file(shard, file.path(output_dir, fname))
        if (verbose) {
            message(sprintf("  shard %d: %.2f GB, %d tensors", shard_idx,
                            shard_size / 1e9, length(shard)))
        }
        shard <<- list()
        shard_size <<- 0
        gc(verbose = FALSE)
    }

    torch::with_no_grad({
        for (key in keys) {
            t <- handle$get_tensor(key)
            bytes <- prod(t$shape) *
            as.integer(switch(as.character(t$dtype),
                              Float = 4L, Double = 8L, Half = 2L,
                              BFloat16 = 2L, Byte = 1L, Char = 1L,
                              Long = 8L, Int = 4L, 4L))
            if (shard_size > 0 && shard_size + bytes > shard_bytes) {
                flush()
            }
            shard[[key]] <- t
            shard_size <- shard_size + bytes
            total_size <- total_size + bytes
            key_to_shard[[key]] <- shard_idx + 1L
        }
    })
    flush()

    n <- shard_idx
    # rename tmp shards to the diffusers -of- convention, build the map
    weight_map <- list()
    final_names <- character(n)
    for (i in seq_len(n)) {
        final_names[i] <- sprintf("%s-%05d-of-%05d.safetensors", base, i, n)
        file.rename(file.path(output_dir, sprintf("%s-%05d.tmp.safetensors", base, i)),
                    file.path(output_dir, final_names[i]))
    }
    for (key in names(key_to_shard)) {
        weight_map[[key]] <- final_names[key_to_shard[[key]]]
    }
    index <- list(metadata = list(total_size = total_size),
                  weight_map = weight_map)
    index_path <- file.path(output_dir, paste0(base, ".safetensors.index.json"))
    jsonlite::write_json(index, index_path, auto_unbox = TRUE, pretty = TRUE)
    if (verbose) {
        message(sprintf("Re-sharded %d tensors into %d shards (%.2f GB): %s",
                        length(keys), n, total_size / 1e9, output_dir))
    }
    invisible(index_path)
}
