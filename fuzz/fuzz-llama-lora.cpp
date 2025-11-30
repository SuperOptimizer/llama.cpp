/*
 * LoRA adapter loading fuzzer for llama.cpp
 *
 * Tests LoRA adapter file parsing:
 * - llama_adapter_lora_init() - Load LoRA adapter from file
 * - Adapter metadata parsing
 *
 * LoRA adapters use GGUF format but have specific structure.
 *
 * Focus areas:
 * - Malformed GGUF structure
 * - Invalid tensor shapes for LoRA
 * - Adapter metadata validation
 */

#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/mman.h>

static int create_memfd(const uint8_t* data, size_t size) {
    int fd = memfd_create("fuzz_llama_lora", MFD_CLOEXEC);
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // Need at least a minimal GGUF header
    if (size < 32) return 0;

    // Cap input size to prevent OOM
    if (size > 5 * 1024 * 1024) return 0;

    llama_log_set(null_log_callback, nullptr);

    int fd = create_memfd(data, size);
    if (fd < 0) return 0;

    char path[64];
    get_fd_path(fd, path, sizeof(path));

    llama_backend_init();

    // Try to load as LoRA adapter (without a base model)
    // This tests the GGUF parsing for LoRA structure
    struct llama_adapter_lora* adapter = llama_adapter_lora_init(nullptr, path);

    if (adapter) {
        // Exercise adapter metadata accessors
        char buf[256];

        // Get metadata count and iterate
        int32_t n_meta = llama_adapter_meta_count(adapter);
        if (n_meta > 0 && n_meta < 1000) {
            for (int32_t i = 0; i < n_meta && i < 100; i++) {
                llama_adapter_meta_key_by_index(adapter, i, buf, sizeof(buf));
                llama_adapter_meta_val_str_by_index(adapter, i, buf, sizeof(buf));
            }
        }

        // Try to get specific metadata
        llama_adapter_meta_val_str(adapter, "general.name", buf, sizeof(buf));
        llama_adapter_meta_val_str(adapter, "adapter.type", buf, sizeof(buf));
        llama_adapter_meta_val_str(adapter, "adapter.lora.alpha", buf, sizeof(buf));

        // Get invocation tokens info
        uint64_t n_inv = llama_adapter_get_alora_n_invocation_tokens(adapter);
        if (n_inv > 0 && n_inv < 1000) {
            const llama_token* inv_tokens = llama_adapter_get_alora_invocation_tokens(adapter);
            (void)inv_tokens;
        }

        llama_adapter_lora_free(adapter);
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
