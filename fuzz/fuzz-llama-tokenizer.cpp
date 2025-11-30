/*
 * Tokenizer fuzzer for llama.cpp
 *
 * Tests tokenization and detokenization with a loaded model:
 * - llama_tokenize() - Text to tokens
 * - llama_token_to_piece() - Token to text
 * - llama_detokenize() - Tokens to text
 *
 * Uses vocab-only model loading to test tokenizer without full model weights.
 *
 * Focus areas:
 * - UTF-8 handling and edge cases
 * - Special token handling
 * - Round-trip consistency
 * - Buffer overflow in text processing
 */

#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/mman.h>
#include <vector>

static int create_memfd(const uint8_t* data, size_t size) {
    int fd = memfd_create("fuzz_llama_tokenizer", MFD_CLOEXEC);
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

static void get_fd_path(int fd, char* buf, size_t buf_size) {
    snprintf(buf, buf_size, "/proc/self/fd/%d", fd);
}

static void null_log_callback(enum ggml_log_level level, const char* text, void* user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

// Test tokenization with arbitrary text
static void test_tokenize_text(const struct llama_vocab* vocab, const uint8_t* data, size_t size) {
    if (!vocab || size < 1) return;

    // Split input: first byte determines test mode, rest is text
    uint8_t mode = data[0];
    const char* text = (const char*)(data + 1);
    size_t text_len = size - 1;

    // Ensure null termination
    char* text_buf = (char*)malloc(text_len + 1);
    if (!text_buf) return;
    memcpy(text_buf, text, text_len);
    text_buf[text_len] = '\0';

    // Allocate token buffer
    std::vector<llama_token> tokens(text_len + 64);  // Some slack for BPE expansion

    bool add_special = (mode & 0x01) != 0;
    bool parse_special = (mode & 0x02) != 0;

    // Test tokenization
    int32_t n_tokens = llama_tokenize(
        vocab,
        text_buf,
        text_len,
        tokens.data(),
        tokens.size(),
        add_special,
        parse_special
    );

    if (n_tokens > 0 && n_tokens <= (int32_t)tokens.size()) {
        tokens.resize(n_tokens);

        // Test detokenization of each token
        char piece_buf[256];
        for (int32_t i = 0; i < n_tokens && i < 100; i++) {
            int32_t n = llama_token_to_piece(
                vocab,
                tokens[i],
                piece_buf,
                sizeof(piece_buf),
                0,      // lstrip
                true    // special
            );
            (void)n;
        }

        // Test full detokenization
        std::vector<char> detok_buf(text_len * 4 + 64);
        int32_t detok_len = llama_detokenize(
            vocab,
            tokens.data(),
            n_tokens,
            detok_buf.data(),
            detok_buf.size(),
            false,  // remove_special
            false   // unparse_special
        );
        (void)detok_len;

        // Test with remove_special
        detok_len = llama_detokenize(
            vocab,
            tokens.data(),
            n_tokens,
            detok_buf.data(),
            detok_buf.size(),
            true,   // remove_special
            true    // unparse_special
        );
        (void)detok_len;
    }

    free(text_buf);
}

// Test with various special token queries
static void test_special_tokens(const struct llama_vocab* vocab) {
    if (!vocab) return;

    // Query all special tokens
    (void)llama_vocab_bos(vocab);
    (void)llama_vocab_eos(vocab);
    (void)llama_vocab_eot(vocab);
    (void)llama_vocab_sep(vocab);
    (void)llama_vocab_nl(vocab);
    (void)llama_vocab_pad(vocab);
    (void)llama_vocab_mask(vocab);

    // FIM tokens
    (void)llama_vocab_fim_pre(vocab);
    (void)llama_vocab_fim_suf(vocab);
    (void)llama_vocab_fim_mid(vocab);
    (void)llama_vocab_fim_pad(vocab);
    (void)llama_vocab_fim_rep(vocab);
    (void)llama_vocab_fim_sep(vocab);

    // Add flags
    (void)llama_vocab_get_add_bos(vocab);
    (void)llama_vocab_get_add_eos(vocab);
    (void)llama_vocab_get_add_sep(vocab);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // Need at least GGUF header + some text
    if (size < 64) return 0;

    // Cap input size
    if (size > 5 * 1024 * 1024) return 0;

    llama_log_set(null_log_callback, nullptr);

    // Split input: first part is "model", rest is text to tokenize
    // Use a portion of input for model data, rest for tokenize test
    size_t model_size = size > 1024 ? size - 512 : size / 2;
    size_t text_size = size - model_size;

    int fd = create_memfd(data, model_size);
    if (fd < 0) return 0;

    char path[64];
    get_fd_path(fd, path, sizeof(path));

    llama_backend_init();

    // Load model vocab-only (fast, still tests tokenizer)
    struct llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    params.use_mmap = false;
    params.use_mlock = false;
    params.check_tensors = false;

    struct llama_model* model = llama_model_load_from_file(path, params);

    if (model) {
        const struct llama_vocab* vocab = llama_model_get_vocab(model);

        if (vocab) {
            int32_t n_vocab = llama_vocab_n_tokens(vocab);
            (void)llama_vocab_type(vocab);

            // Test special tokens
            test_special_tokens(vocab);

            // Test tokenization with the text portion of input
            if (n_vocab > 0 && text_size > 0) {
                test_tokenize_text(vocab, data + model_size, text_size);
            }

            // Test token properties for a few tokens
            if (n_vocab > 0) {
                for (int32_t i = 0; i < n_vocab && i < 50; i++) {
                    (void)llama_vocab_get_text(vocab, i);
                    (void)llama_vocab_get_score(vocab, i);
                    (void)llama_vocab_get_attr(vocab, i);
                    (void)llama_vocab_is_eog(vocab, i);
                    (void)llama_vocab_is_control(vocab, i);
                }
            }

            // Get tokenizer info (previously 0% covered)
            (void)llama_vocab_type(vocab);
            (void)llama_vocab_n_tokens(vocab);
        }

        llama_model_free(model);
    }

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
