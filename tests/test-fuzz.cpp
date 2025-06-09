#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <sstream>
#include <random>
#include "llama.h"
#include "common.h"
#include "sampling.h"
#include <cstring>
#include <fstream>

#ifdef __has_feature
#if __has_feature(memory_sanitizer)
#include <sanitizer/msan_interface.h>
#define MSAN_ENABLED 1
#endif
#endif

#ifndef MSAN_ENABLED
#define MSAN_ENABLED 0
#endif

#ifndef FUZZING_UNIQUE
#define FUZZING_UNIQUE ""
#endif


#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 32 || size > 100 * 1024 * 1024) return 0;
    srand(time(0));

#ifdef  FUZZING_UNSTABLE
    printf("unstable mode\n");
    if (rand()%10==0) {
        size -= rand() % (size-1);
    }
#endif

#if MSAN_ENABLED
    __msan_unpoison(data, size);
#endif

    char temp_template[64] = {'\0'};
    strncpy(temp_template, "/tmp/fuzz_model_XXXXXX", 63);

    int fd = mkstemp(temp_template);
    if (fd == -1) {
        printf("failed to mkstemp\n");
        return 0;
    }

    ssize_t written = write(fd, data, size);
    close(fd);
    if (written != (ssize_t)size) {
        unlink(temp_template);
        return 0;
    }

    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    common_sampler* smpl = nullptr;

    try {
        llama_backend_init();

        llama_model_params model_params = llama_model_default_params();
        model_params.use_mmap = true;
        model_params.progress_callback = nullptr;

        model = llama_load_model_from_file(temp_template, model_params);
        if (model) {
            llama_context_params ctx_params = llama_context_default_params();
            ctx_params.n_ctx = 32;

            ctx = llama_new_context_with_model(model, ctx_params);
            if (ctx) {
                const std::string prompt = "Hello";
                const llama_vocab* vocab = llama_model_get_vocab(model);
                std::vector<llama_token> tokens(25);

                // Tokenize initial prompt
                int32_t n_tokens = llama_tokenize(
                    vocab,
                    prompt.c_str(),
                    static_cast<int32_t>(prompt.length()),
                    tokens.data(),
                    static_cast<int32_t>(tokens.size()),
                    true,
                    true
                );
                if (n_tokens > 0 && n_tokens < 32) {
                    tokens.resize(n_tokens);

                    // Process initial prompt
                    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
                    int result = llama_decode(ctx, batch);
                    if (result == 0) {
                        common_params params;
                        // Deterministic sampling settings
                        params.sampling.seed = 12345;           // Fixed seed
                        params.sampling.temp = 0.0f;            // Greedy sampling (most deterministic)
                        params.sampling.top_k = 1;              // Only consider top token
                        params.sampling.top_p = 1.0f;           // No nucleus sampling
                        params.sampling.penalty_repeat = 1.0f;  // No repetition penalty
                        params.sampling.penalty_freq = 0.0f;    // No frequency penalty
                        params.sampling.penalty_present = 0.0f; // No presence penalty
                        params.sampling.mirostat = 0;           // Disable mirostat
                        params.sampling.dynatemp_range = 0.0f;  // Disable dynamic temperature

                        smpl = common_sampler_init(model, params.sampling);
                        if (smpl) {
                            std::vector<llama_token> generated_tokens;
                            std::string generated_text;

                            // Generate tokens to fill context
                            int max_new_tokens = 32 - n_tokens; // Fill remaining context
                            for (int i = 0; i < max_new_tokens; i++) {
                                llama_token new_token = common_sampler_sample(smpl, ctx, -1);

                                // Check for EOS token
                                if (llama_vocab_is_eog(vocab, new_token)) {
                                    printf("Hit EOS token\n");
                                    break;
                                }

                                generated_tokens.push_back(new_token);
                                common_sampler_accept(smpl, new_token, true);

                                // Process the new token
                                llama_batch single_batch = llama_batch_get_one(&new_token, 1);
                                result = llama_decode(ctx, single_batch);
                                if (result != 0) {
                                    printf("Decode failed at token %d\n", i);
                                    break;
                                }

                                // Convert token to text and accumulate
                                const char* token_str = llama_vocab_get_text(vocab, new_token);
                                if (token_str) {
                                    generated_text += token_str;
                                }
                            }

                            printf("=== GENERATED TEXT ===\n");
                            printf("Prompt: %s\n", prompt.c_str());
                            printf("Generated (%d tokens): %s\n", (int)generated_tokens.size(), generated_text.c_str());
                            printf("=== END ===\n");
                        }
                    }
                }
            }
        }
    } catch (const std::exception& e) {
         printf("Exception: %s\n", e.what());
         if (smpl) common_sampler_free(smpl);
         if (ctx) llama_free(ctx);
         if (model) llama_free_model(model);
         llama_backend_free();
         unlink(temp_template);
         throw; // re-raise with full details preserved
     } catch (...) {
         printf("Unknown exception\n");
         if (smpl) common_sampler_free(smpl);
         if (ctx) llama_free(ctx);
         if (model) llama_free_model(model);
         llama_backend_free();
         unlink(temp_template);
         throw; // re-raise original exception
     }

    if (smpl) common_sampler_free(smpl);
    if (ctx) llama_free(ctx);
    if (model) llama_free_model(model);
    llama_backend_free();

    unlink(temp_template);
    printf("executed successfully\n");
    return 0;
}

