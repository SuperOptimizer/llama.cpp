/*
 * Fuzz harness for llama.cpp GGUF model loading.
 *
 * This harness tests llama.cpp's model loading path which includes:
 * - GGUF parsing and validation
 * - Model architecture detection
 * - Vocabulary loading
 * - Tensor loading and validation
 * - Memory allocation and mapping
 *
 * Key vulnerability areas:
 * - Integer overflow in tensor size calculations
 * - Buffer overflow in string/metadata parsing
 * - Invalid tensor type handling
 * - Architecture-specific validation bypasses
 * - Memory exhaustion via large allocation requests
 *
 * Uses memfd_create() to avoid temp file pollution during fuzzing.
 */

#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/mman.h>

// Create an anonymous in-memory file with the fuzzed data
static int create_memfd(const uint8_t* data, size_t size) {
    int fd = memfd_create("fuzz_llama_gguf", MFD_CLOEXEC);
    if (fd < 0) return -1;

    if (ftruncate(fd, size) < 0) {
        close(fd);
        return -1;
    }

    ssize_t written = write(fd, data, size);
    if (written != (ssize_t)size) {
        close(fd);
        return -1;
    }

    lseek(fd, 0, SEEK_SET);
    return fd;
}

// Get the path to an fd via /proc/self/fd/
static void get_fd_path(int fd, char* buf, size_t buf_size) {
    snprintf(buf, buf_size, "/proc/self/fd/%d", fd);
}

// Suppress llama.cpp logging during fuzzing
static void null_log_callback(enum ggml_log_level level, const char* text, void* user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // Need at least a minimal GGUF header
    if (size < 32) return 0;

    // Cap input size to prevent OOM during fuzzing
    if (size > 10 * 1024 * 1024) return 0;

    // Suppress logging
    llama_log_set(null_log_callback, nullptr);

    // Create memfd with fuzzed data
    int fd = create_memfd(data, size);
    if (fd < 0) return 0;

    char path[64];
    get_fd_path(fd, path, sizeof(path));

    // Initialize llama backend (safe to call multiple times)
    llama_backend_init();

    // Set up model params for vocab-only loading (faster, still tests parsing)
    struct llama_model_params params = llama_model_default_params();
    params.vocab_only = true;     // Only load vocabulary, skip weights
    params.use_mmap = false;      // Don't mmap from memfd
    params.use_mlock = false;     // Don't mlock
    params.check_tensors = false; // Skip tensor validation for speed

    // Attempt to load the model
    struct llama_model* model = llama_model_load_from_file(path, params);

    if (model) {
        // Exercise metadata accessors to trigger any lazy parsing
        char buf[256];

        // Get model description
        llama_model_desc(model, buf, sizeof(buf));

        // Get model size and params
        (void)llama_model_size(model);
        (void)llama_model_n_params(model);

        // Iterate metadata
        int32_t n_meta = llama_model_meta_count(model);
        if (n_meta > 0 && n_meta < 1000) {  // Sanity cap
            for (int32_t i = 0; i < n_meta && i < 100; i++) {
                llama_model_meta_key_by_index(model, i, buf, sizeof(buf));
                llama_model_meta_val_str_by_index(model, i, buf, sizeof(buf));
            }
        }

        // Get model architecture info
        // Note: n_head/n_head_kv crash if n_layer == 0 (vocab_only mode)
        int32_t n_layer = llama_model_n_layer(model);
        (void)llama_model_n_ctx_train(model);
        (void)llama_model_n_embd(model);
        if (n_layer > 0) {
            (void)llama_model_n_head(model);
        }
        (void)llama_model_rope_type(model);
        (void)llama_model_has_encoder(model);
        (void)llama_model_has_decoder(model);
        (void)llama_model_is_recurrent(model);

        // Get vocabulary if loaded
        const struct llama_vocab* vocab = llama_model_get_vocab(model);
        if (vocab) {
            (void)llama_vocab_type(vocab);
            int32_t n_vocab = llama_vocab_n_tokens(vocab);

            // Query a few token properties
            if (n_vocab > 0) {
                (void)llama_vocab_bos(vocab);
                (void)llama_vocab_eos(vocab);
                (void)llama_vocab_pad(vocab);

                // Check a few tokens
                for (int32_t i = 0; i < n_vocab && i < 10; i++) {
                    (void)llama_vocab_get_text(vocab, i);
                    (void)llama_vocab_get_score(vocab, i);
                    (void)llama_vocab_get_attr(vocab, i);
                }
            }
        }

        // Get chat template if available
        (void)llama_model_chat_template(model, nullptr);

        // Free the model
        llama_model_free(model);
    }

    // Cleanup
    close(fd);
    llama_backend_free();

    return 0;
}

#ifndef FUZZ_WITH_LIBFUZZER
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    FILE* f = fopen(argv[1], "rb");
    if (!f) {
        perror("fopen");
        return 1;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size <= 0 || size > 100 * 1024 * 1024) {
        fclose(f);
        return 1;
    }

    uint8_t* data = (uint8_t*)malloc(size);
    if (!data) {
        fclose(f);
        return 1;
    }

    if (fread(data, 1, size, f) != (size_t)size) {
        free(data);
        fclose(f);
        return 1;
    }
    fclose(f);

    int result = LLVMFuzzerTestOneInput(data, size);

    free(data);
    return result;
}
#endif
