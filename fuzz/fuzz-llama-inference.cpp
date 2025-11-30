/*
 * Fuzz harness for llama.cpp full model loading and inference.
 *
 * Unlike fuzz-llama-gguf which only does vocab_only loading, this harness:
 * - Loads the full model with tensors
 * - Creates a context
 * - Runs a simple inference pass
 *
 * This tests the full inference pipeline including:
 * - Tensor allocation and loading
 * - Architecture-specific model building (src/models/*.cpp)
 * - KV cache initialization
 * - Forward pass computation
 * - Sampling
 *
 * llama-context.cpp coverage targets:
 * - State save/load: llama_state_get_size, llama_state_get_data, llama_state_set_data
 * - Sequence state: llama_state_seq_get_size, llama_state_seq_get_data, llama_state_seq_set_data
 * - Threading: llama_set_n_threads, llama_n_threads, llama_n_threads_batch
 * - Memory operations: llama_memory_seq_rm, llama_memory_seq_cp, llama_memory_seq_keep, etc.
 * - Encode: llama_encode (for encoder models)
 * - Performance: llama_perf_context, llama_perf_context_reset
 * - Embeddings: llama_get_embeddings_ith, llama_get_embeddings_seq
 *
 * Exceptions are NOT caught - we want the fuzzer to find crashes.
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
    int fd = memfd_create("fuzz_llama_inference", MFD_CLOEXEC);
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

static bool g_verbose = false;

static void null_log_callback(enum ggml_log_level level, const char* text, void* user_data) {
    (void)user_data;
    if (g_verbose) {
        fprintf(stderr, "[%d] %s", level, text);
    }
}

// Query model metadata to exercise model loading code
static void query_model_metadata(struct llama_model* model) {
    // These exercise various model accessor functions
    (void)llama_model_n_params(model);
    (void)llama_model_size(model);
    (void)llama_model_n_ctx_train(model);
    (void)llama_model_n_embd(model);
    (void)llama_model_n_layer(model);
    (void)llama_model_n_head(model);

    // Get model description
    char buf[256];
    llama_model_desc(model, buf, sizeof(buf));

    // Check if model has encoder/decoder
    (void)llama_model_has_encoder(model);
    (void)llama_model_has_decoder(model);

    // Rope parameters
    (void)llama_model_rope_freq_scale_train(model);
}

// Test state save/load functionality (llama-context.cpp coverage)
static void test_state_save_load(struct llama_context* ctx) {
    // Get state size
    size_t state_size = llama_state_get_size(ctx);
    if (state_size == 0 || state_size > 100 * 1024 * 1024) return;

    // Allocate buffer and save state
    std::vector<uint8_t> state_buf(state_size);
    size_t saved = llama_state_get_data(ctx, state_buf.data(), state_buf.size());
    if (saved == 0) return;

    // Restore state
    size_t restored = llama_state_set_data(ctx, state_buf.data(), state_buf.size());
    (void)restored;
}

// Test sequence state save/load (llama-context.cpp coverage)
static void test_seq_state_save_load(struct llama_context* ctx, llama_seq_id seq_id) {
    // Get sequence state size
    size_t seq_size = llama_state_seq_get_size(ctx, seq_id);
    if (seq_size == 0 || seq_size > 10 * 1024 * 1024) return;

    // Save sequence state
    std::vector<uint8_t> seq_buf(seq_size);
    size_t saved = llama_state_seq_get_data(ctx, seq_buf.data(), seq_buf.size(), seq_id);
    if (saved == 0) return;

    // Restore sequence state to a different sequence
    llama_seq_id dest_seq = (seq_id + 1) % 2;
    size_t restored = llama_state_seq_set_data(ctx, seq_buf.data(), seq_buf.size(), dest_seq);
    (void)restored;
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

// Test memory sequence operations (llama-context.cpp coverage)
static void test_memory_seq_operations(llama_memory_t mem, llama_seq_id seq_id) {
    if (!mem) return;

    // Query sequence position bounds
    llama_pos pos_min = llama_memory_seq_pos_min(mem, seq_id);
    llama_pos pos_max = llama_memory_seq_pos_max(mem, seq_id);
    (void)pos_min;
    (void)pos_max;

    // Copy sequence to another
    llama_seq_id dest_seq = (seq_id + 1) % 4;
    llama_memory_seq_cp(mem, seq_id, dest_seq, 0, pos_max);

    // Add position offset to sequence
    llama_memory_seq_add(mem, dest_seq, 0, pos_max, 1);

    // Divide positions in sequence
    llama_memory_seq_div(mem, dest_seq, 0, pos_max, 2);

    // Remove part of a sequence
    llama_memory_seq_rm(mem, dest_seq, 0, pos_max / 2);

    // Keep only specific sequence (clears others)
    // Only do this on a non-primary sequence to avoid clearing test data
    if (dest_seq != seq_id) {
        // We skip llama_memory_seq_keep to avoid clearing our test sequence
    }
}

// Test performance counters (llama-context.cpp coverage)
static void test_perf_counters(struct llama_context* ctx) {
    // Get performance data
    struct llama_perf_context_data perf = llama_perf_context(ctx);
    (void)perf.t_start_ms;
    (void)perf.t_load_ms;
    (void)perf.t_p_eval_ms;
    (void)perf.t_eval_ms;
    (void)perf.n_p_eval;
    (void)perf.n_eval;

    // Reset performance counters
    llama_perf_context_reset(ctx);
}

// Test encode for encoder models (llama-context.cpp coverage)
static void test_encode(struct llama_context* ctx, struct llama_model* model,
                        const struct llama_vocab* vocab, int32_t n_vocab) {
    // Only test encode if model has encoder
    if (!llama_model_has_encoder(model)) return;

    llama_token bos = vocab ? llama_vocab_bos(vocab) : 0;
    if (bos < 0 || bos >= n_vocab) bos = 0;

    // Create a batch for encoding
    struct llama_batch batch = llama_batch_init(4, 0, 1);
    batch.n_tokens = 1;
    batch.token[0] = bos;
    batch.pos[0] = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = false;

    // Run encode
    int ret = llama_encode(ctx, batch);
    (void)ret;

    llama_batch_free(batch);
}

// Test embeddings functions (llama-context.cpp coverage)
static void test_embeddings(struct llama_context* ctx) {
    // Get embeddings at specific position
    float* embd_ith = llama_get_embeddings_ith(ctx, 0);
    (void)embd_ith;

    // Get embeddings for sequence
    float* embd_seq = llama_get_embeddings_seq(ctx, 0);
    (void)embd_seq;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // Need at least a minimal GGUF header
    if (size < 32) return 0;

    // Cap input size to prevent OOM - allow larger files for valid models
    if (size > 20 * 1024 * 1024) return 0;

    llama_log_set(null_log_callback, nullptr);

    int fd = create_memfd(data, size);
    if (fd < 0) return 0;

    char path[64];
    get_fd_path(fd, path, sizeof(path));

    llama_backend_init();

    // Full model loading (not vocab_only)
    struct llama_model_params model_params = llama_model_default_params();
    model_params.vocab_only = false;    // Load full model with tensors
    model_params.use_mmap = false;
    model_params.use_mlock = false;
    model_params.check_tensors = false;  // Skip validation for fuzzing - fuzzer will mutate data

    struct llama_model* model = llama_model_load_from_file(path, model_params);

    if (model) {
        // Query metadata to exercise model code paths
        query_model_metadata(model);

        // Create context for inference
        struct llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 64;          // Small context for fuzzing
        ctx_params.n_batch = 8;
        ctx_params.n_ubatch = 8;
        ctx_params.n_threads = 1;
        ctx_params.n_threads_batch = 1;
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

        struct llama_context* ctx = llama_init_from_model(model, ctx_params);

        if (ctx) {
            const struct llama_vocab* vocab = llama_model_get_vocab(model);
            int32_t n_vocab = vocab ? llama_vocab_n_tokens(vocab) : 0;

            // Exercise context property queries (previously 0% covered)
            (void)llama_n_ctx(ctx);
            (void)llama_n_batch(ctx);
            (void)llama_n_ubatch(ctx);
            (void)llama_n_seq_max(ctx);
            (void)llama_pooling_type(ctx);

            if (n_vocab > 0 && n_vocab < 1000000) {
                // Create a simple batch with BOS token
                struct llama_batch batch = llama_batch_init(8, 0, 1);

                llama_token bos = vocab ? llama_vocab_bos(vocab) : 0;
                if (bos < 0 || bos >= n_vocab) bos = 0;

                // Add a single token
                batch.n_tokens = 1;
                batch.token[0] = bos;
                batch.pos[0] = 0;
                batch.n_seq_id[0] = 1;
                batch.seq_id[0][0] = 0;
                batch.logits[0] = true;

                // Run inference (decode)
                int ret = llama_decode(ctx, batch);

                if (ret == 0) {
                    // Get logits using both methods (llama_get_logits_ith was 0% covered)
                    float* logits = llama_get_logits(ctx);
                    float* logits_ith = llama_get_logits_ith(ctx, 0);
                    (void)logits_ith;

                    if (logits && n_vocab > 0) {
                        // Simple greedy sampling - find max logit
                        float max_logit = logits[0];
                        llama_token max_token = 0;
                        int32_t limit = (n_vocab < 1000) ? n_vocab : 1000;
                        for (int32_t i = 1; i < limit; i++) {
                            if (logits[i] > max_logit) {
                                max_logit = logits[i];
                                max_token = i;
                            }
                        }
                        (void)max_token;
                    }

                    // Exercise KV cache (memory) operations (previously 0% covered)
                    llama_memory_t mem = llama_get_memory(ctx);
                    if (mem) {
                        llama_memory_seq_pos_max(mem, 0);

                        // Test memory sequence operations for coverage
                        test_memory_seq_operations(mem, 0);
                    }

                    // Test threading controls
                    test_threading(ctx);

                    // Test performance counters
                    test_perf_counters(ctx);

                    // Test embeddings functions (multiple methods)
                    test_embeddings(ctx);

                    // Try embeddings if model supports pooling (previously 0% covered)
                    if (llama_pooling_type(ctx) != LLAMA_POOLING_TYPE_NONE) {
                        float* embd = llama_get_embeddings(ctx);
                        (void)embd;
                    }

                    // Test state save/load after we have some state
                    test_state_save_load(ctx);

                    // Test sequence state save/load
                    test_seq_state_save_load(ctx, 0);

                    // Test DRY sampler (requires vocab, previously 0% covered)
                    if (vocab) {
                        // Create sequence breakers for DRY
                        const char* seq_breakers[] = {".", "!", "?", "\n"};
                        int32_t n_ctx_train = llama_model_n_ctx_train(model);
                        if (n_ctx_train <= 0) n_ctx_train = 2048;

                        struct llama_sampler* dry = llama_sampler_init_dry(
                            vocab,
                            n_ctx_train,
                            1.75f,    // dry_multiplier
                            2.0f,     // dry_base
                            2,        // dry_allowed_length
                            128,      // dry_penalty_last_n
                            seq_breakers,
                            4         // num_breakers
                        );

                        if (dry) {
                            // Create a simple sampler chain with DRY
                            struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
                            chain_params.no_perf = true;
                            struct llama_sampler* chain = llama_sampler_chain_init(chain_params);

                            if (chain) {
                                llama_sampler_chain_add(chain, dry);

                                // Create token data for sampling
                                int32_t test_vocab = (n_vocab < 100) ? n_vocab : 100;
                                std::vector<llama_token_data> candidates(test_vocab);
                                for (int32_t j = 0; j < test_vocab; j++) {
                                    candidates[j].id = j;
                                    candidates[j].logit = logits ? logits[j] : 0.0f;
                                    candidates[j].p = 0.0f;
                                }

                                llama_token_data_array cur_p = {
                                    candidates.data(),
                                    (size_t)test_vocab,
                                    -1,
                                    false
                                };

                                llama_sampler_apply(chain, &cur_p);
                                llama_sampler_free(chain);
                            } else {
                                llama_sampler_free(dry);
                            }
                        }
                    }
                }

                // Exercise causal attention toggle (previously 0% covered)
                llama_set_causal_attn(ctx, false);
                llama_set_causal_attn(ctx, true);

                // Clear KV cache (memory) and do another decode (previously 0% covered)
                llama_memory_t mem2 = llama_get_memory(ctx);
                if (mem2) {
                    llama_memory_clear(mem2, true);
                }

                // Second decode after cache clear
                ret = llama_decode(ctx, batch);
                (void)ret;

                llama_batch_free(batch);
            }

            // Test encode for encoder models (outside batch scope)
            test_encode(ctx, model, vocab, n_vocab);

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
        fprintf(stderr, "Usage: %s [-v] <input_file>\n", argv[0]);
        return 1;
    }

    int arg_idx = 1;
    if (argc > 2 && strcmp(argv[1], "-v") == 0) {
        g_verbose = true;
        arg_idx = 2;
    }

    FILE* f = fopen(argv[arg_idx], "rb");
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
