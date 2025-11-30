/*
 * Grammar (GBNF) parser fuzzer for llama.cpp
 *
 * Tests the GBNF grammar parsing functionality:
 * - llama_grammar_parser::parse()
 * - Grammar validation and compilation
 * - UTF-8 handling in grammar rules
 *
 * Focus areas:
 * - Malformed grammar syntax
 * - Unicode escape sequences (\x, \u, \U)
 * - Deeply nested rules
 * - Recursive grammar definitions
 * - Character class edge cases ([a-z], [^...])
 */

#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

// Suppress logging during fuzzing
static void null_log_callback(enum ggml_log_level level, const char* text, void* user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // Need at least some grammar content
    if (size < 4) return 0;

    // Cap input size to prevent excessive parsing time
    if (size > 64 * 1024) return 0;

    // Suppress logging
    llama_log_set(null_log_callback, nullptr);

    // Create null-terminated string from fuzz input
    char* grammar_str = (char*)malloc(size + 1);
    if (!grammar_str) return 0;
    memcpy(grammar_str, data, size);
    grammar_str[size] = '\0';

    // Initialize llama backend
    llama_backend_init();

    // Try to create a grammar sampler with the fuzzed input
    // This will parse the GBNF grammar string
    // Grammar parsing can throw exceptions on malformed input
    try {
        struct llama_sampler* sampler = llama_sampler_init_grammar(
            nullptr,     // vocab (nullptr for testing)
            grammar_str, // grammar string
            "root"       // root rule name
        );

        if (sampler) {
            // Exercise the sampler a bit
            const char* name = llama_sampler_name(sampler);
            (void)name;

            // Reset and free
            llama_sampler_reset(sampler);
            llama_sampler_free(sampler);
        }
    } catch (const std::exception&) {
        // Expected for malformed grammars
    } catch (...) {
        // Catch any other exceptions
    }

    // Also try with different root rule names that might be in the grammar
    const char* common_roots[] = {"start", "main", "expr", "statement", "value"};
    for (int i = 0; i < 5; i++) {
        try {
            struct llama_sampler* sampler2 = llama_sampler_init_grammar(
                nullptr,
                grammar_str,
                common_roots[i]
            );
            if (sampler2) {
                llama_sampler_free(sampler2);
            }
        } catch (...) {
            // Expected for invalid root rules
        }
    }

    free(grammar_str);
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
