/*
 * KV Cache manipulation fuzzer for llama.cpp
 *
 * Tests the memory (KV cache) manipulation functions:
 * - llama_memory_clear() - Clear cache contents
 * - llama_memory_seq_rm() - Remove tokens from sequence
 * - llama_memory_seq_cp() - Copy sequence
 * - llama_memory_seq_keep() - Keep only specified sequence
 * - llama_memory_seq_add() - Shift positions
 * - llama_memory_seq_div() - Divide positions
 * - llama_memory_seq_pos_min/max() - Query positions
 *
 * Also tests llama-context.cpp state functions:
 * - llama_state_seq_get_size/get_data/set_data - Sequence state save/load
 * - llama_state_seq_get_size_ext/get_data_ext/set_data_ext - Extended API
 * - llama_set_n_threads/llama_n_threads - Threading controls
 * - llama_perf_context/reset - Performance counters
 *
 * These functions were previously at 0% coverage.
 * This harness loads a model, performs some decodes to populate the cache,
 * then exercises the various cache manipulation operations.
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
    int fd = memfd_create("fuzz_llama_kv_cache", MFD_CLOEXEC);
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

class FuzzInput {
public:
    FuzzInput(const uint8_t* data, size_t size)
        : data_(data), size_(size), pos_(0) {}

    uint8_t u8() {
        return pos_ < size_ ? data_[pos_++] : 0;
    }

    int32_t i32() {
        int32_t v = 0;
        if (pos_ + 4 <= size_) {
            memcpy(&v, data_ + pos_, 4);
            pos_ += 4;
        }
        return v;
    }

    bool has_data() const { return pos_ < size_; }
    size_t remaining() const { return size_ > pos_ ? size_ - pos_ : 0; }
    size_t consumed() const { return pos_; }

    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }

private:
    const uint8_t* data_;
    size_t size_;
    size_t pos_;
};

// Test cache manipulation operations
static void test_cache_operations(struct llama_context* ctx, FuzzInput& in) {
    llama_memory_t mem = llama_get_memory(ctx);
    if (!mem) return;

    // Get current cache state
    llama_pos pos_min_0 = llama_memory_seq_pos_min(mem, 0);
    llama_pos pos_max_0 = llama_memory_seq_pos_max(mem, 0);
    (void)pos_min_0;
    (void)pos_max_0;

    // Check if memory supports shifting
    bool can_shift = llama_memory_can_shift(mem);
    (void)can_shift;

    // Run a series of operations based on fuzz input
    int n_ops = in.u8() % 10 + 1;
    for (int op = 0; op < n_ops && in.has_data(); op++) {
        uint8_t op_type = in.u8() % 8;
        llama_seq_id seq_id = in.u8() % 4;  // Use small number of sequences
        llama_pos p0 = in.i32() % 64;       // Small positions for the small context
        llama_pos p1 = in.i32() % 64;

        // Ensure p0 <= p1 when both are positive
        if (p0 > 0 && p1 > 0 && p0 > p1) {
            llama_pos tmp = p0;
            p0 = p1;
            p1 = tmp;
        }

        switch (op_type) {
            case 0: {
                // llama_memory_seq_rm - Remove tokens from sequence
                bool result = llama_memory_seq_rm(mem, seq_id, p0, p1);
                (void)result;
                break;
            }

            case 1: {
                // llama_memory_seq_cp - Copy sequence to another
                llama_seq_id dst_seq = (seq_id + 1) % 4;
                llama_memory_seq_cp(mem, seq_id, dst_seq, p0, p1);
                break;
            }

            case 2: {
                // llama_memory_seq_keep - Keep only specified sequence
                llama_memory_seq_keep(mem, seq_id);
                break;
            }

            case 3: {
                // llama_memory_seq_add - Shift positions
                llama_pos delta = (in.i32() % 20) - 10;  // -10 to +10
                llama_memory_seq_add(mem, seq_id, p0, p1, delta);
                break;
            }

            case 4: {
                // llama_memory_seq_div - Divide positions
                int d = (in.u8() % 4) + 1;  // 1 to 4
                llama_memory_seq_div(mem, seq_id, p0, p1, d);
                break;
            }

            case 5: {
                // llama_memory_seq_pos_min - Query min position
                llama_pos pos = llama_memory_seq_pos_min(mem, seq_id);
                (void)pos;
                break;
            }

            case 6: {
                // llama_memory_seq_pos_max - Query max position
                llama_pos pos = llama_memory_seq_pos_max(mem, seq_id);
                (void)pos;
                break;
            }

            case 7: {
                // llama_memory_clear - Clear cache
                bool clear_data = (in.u8() & 1) != 0;
                llama_memory_clear(mem, clear_data);
                break;
            }
        }
    }
}

// Test sequence state save/load (llama-context.cpp coverage)
static void test_seq_state(struct llama_context* ctx, FuzzInput& in) {
    llama_seq_id seq_id = in.u8() % 4;

    // Test basic sequence state API
    size_t seq_size = llama_state_seq_get_size(ctx, seq_id);
    if (seq_size == 0 || seq_size > 10 * 1024 * 1024) return;

    std::vector<uint8_t> seq_buf(seq_size);
    size_t saved = llama_state_seq_get_data(ctx, seq_buf.data(), seq_buf.size(), seq_id);
    if (saved == 0) return;

    // Restore to a different sequence
    llama_seq_id dest_seq = (seq_id + 1) % 4;
    size_t restored = llama_state_seq_set_data(ctx, seq_buf.data(), seq_buf.size(), dest_seq);
    (void)restored;

    // Test extended sequence state API with flags
    llama_state_seq_flags flags = 0;  // No special flags
    size_t ext_size = llama_state_seq_get_size_ext(ctx, seq_id, flags);
    if (ext_size > 0 && ext_size <= 10 * 1024 * 1024) {
        std::vector<uint8_t> ext_buf(ext_size);
        size_t ext_saved = llama_state_seq_get_data_ext(ctx, ext_buf.data(), ext_buf.size(), seq_id, flags);
        if (ext_saved > 0) {
            llama_seq_id dest_seq2 = (seq_id + 2) % 4;
            llama_state_seq_set_data_ext(ctx, ext_buf.data(), ext_buf.size(), dest_seq2, flags);
        }
    }
}

// Test threading controls (llama-context.cpp coverage)
static void test_threading(struct llama_context* ctx) {
    // Query current thread counts
    int32_t n_threads = llama_n_threads(ctx);
    int32_t n_threads_batch = llama_n_threads_batch(ctx);
    (void)n_threads;
    (void)n_threads_batch;

    // Set thread counts
    llama_set_n_threads(ctx, 1, 1);
    llama_set_n_threads(ctx, 2, 2);
    llama_set_n_threads(ctx, 1, 1);  // Reset to safe values
}

// Test performance counters (llama-context.cpp coverage)
static void test_perf(struct llama_context* ctx) {
    struct llama_perf_context_data perf = llama_perf_context(ctx);
    (void)perf.t_start_ms;
    (void)perf.n_eval;

    llama_perf_context_reset(ctx);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // Need at least GGUF header + some control data
    if (size < 64) return 0;

    // Cap input size
    if (size > 20 * 1024 * 1024) return 0;

    llama_log_set(null_log_callback, nullptr);

    // Parse control bytes from end of input, model data from beginning
    size_t control_size = size > 256 ? 256 : size / 4;
    size_t model_size = size - control_size;

    FuzzInput in(data + model_size, control_size);

    int fd = create_memfd(data, model_size);
    if (fd < 0) return 0;

    char path[64];
    get_fd_path(fd, path, sizeof(path));

    llama_backend_init();

    // Load full model (need tensors for KV cache)
    struct llama_model_params model_params = llama_model_default_params();
    model_params.vocab_only = false;
    model_params.use_mmap = false;
    model_params.use_mlock = false;
    model_params.check_tensors = false;

    struct llama_model* model = llama_model_load_from_file(path, model_params);

    if (model) {
        // Create context with KV cache
        struct llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 64;          // Small context
        ctx_params.n_batch = 8;
        ctx_params.n_ubatch = 8;
        ctx_params.n_threads = 1;
        ctx_params.n_threads_batch = 1;
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        ctx_params.n_seq_max = 4;       // Allow multiple sequences

        struct llama_context* ctx = llama_init_from_model(model, ctx_params);

        if (ctx) {
            const struct llama_vocab* vocab = llama_model_get_vocab(model);
            int32_t n_vocab = vocab ? llama_vocab_n_tokens(vocab) : 0;

            if (n_vocab > 0 && n_vocab < 1000000) {
                // Create a batch and do some decodes to populate the cache
                struct llama_batch batch = llama_batch_init(8, 0, 4);

                llama_token bos = vocab ? llama_vocab_bos(vocab) : 0;
                if (bos < 0 || bos >= n_vocab) bos = 0;

                // Add tokens to multiple sequences to test seq operations
                batch.n_tokens = 0;
                for (int seq = 0; seq < 2 && batch.n_tokens < 6; seq++) {
                    for (int i = 0; i < 3; i++) {
                        int idx = batch.n_tokens;
                        llama_token tok = (bos + i) % n_vocab;
                        batch.token[idx] = tok;
                        batch.pos[idx] = i;
                        batch.n_seq_id[idx] = 1;
                        batch.seq_id[idx][0] = seq;
                        batch.logits[idx] = (i == 2);  // Request logits for last token
                        batch.n_tokens++;
                    }
                }

                // Decode to populate cache
                int ret = llama_decode(ctx, batch);

                if (ret == 0) {
                    // Test threading controls
                    test_threading(ctx);

                    // Test performance counters
                    test_perf(ctx);

                    // Now exercise cache manipulation
                    test_cache_operations(ctx, in);

                    // Test sequence state save/load
                    test_seq_state(ctx, in);

                    // Do another decode after cache manipulation
                    batch.n_tokens = 1;
                    batch.token[0] = bos;
                    batch.pos[0] = 10;  // Later position
                    batch.n_seq_id[0] = 1;
                    batch.seq_id[0][0] = 0;
                    batch.logits[0] = true;

                    ret = llama_decode(ctx, batch);
                    (void)ret;
                }

                llama_batch_free(batch);
            }

            llama_free(ctx);
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
