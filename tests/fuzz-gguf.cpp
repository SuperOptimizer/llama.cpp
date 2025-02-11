// GGUF format fuzzer for llama.cpp
// Compile with: clang++ -g -O1 -fsanitize=fuzzer,address gguf_fuzzer.cpp -I../llama.cpp

#include "llama-model-loader.h"
#include "ggml.h"
#include "llama-chat.h"
#include <cstdint>
#include <cstring>
#include <memory>
#include <filesystem>
#include <random>
#include <stdio.h>
#include <set>

// Global cleanup handler
class GlobalCleanup {
public:
    void track_file(const std::filesystem::path& p) {
        active_files.insert(p);
    }

    void untrack_file(const std::filesystem::path& p) {
        active_files.erase(p);
    }

    ~GlobalCleanup() {
        for (const auto& p : active_files) {
            std::filesystem::remove(p);
        }
    }
private:
    std::set<std::filesystem::path> active_files;
};

static GlobalCleanup cleanup;

class FuzzFile {
public:
    FuzzFile(const uint8_t* data, size_t size) {
        // Create random filename
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 35); // 0-9, a-z
        std::string random_str;
        random_str.reserve(16);
        for (int i = 0; i < 16; i++) {
            int r = dis(gen);
            random_str += (r < 10) ? ('0' + r) : ('a' + (r-10));
        }
        tmp_path = std::filesystem::temp_directory_path() / ("fuzz-" + random_str + ".gguf");
        cleanup.track_file(tmp_path);

        // Write data to temp file
        FILE* out = fopen(tmp_path.c_str(), "wb");
        if (!out || fwrite(data, 1, size, out) != size) {
            if (out) fclose(out);
            std::filesystem::remove(tmp_path);
            cleanup.untrack_file(tmp_path);
            throw std::runtime_error("Failed to write temp file");
        }
        fclose(out);

        // Open for reading
        fp = fopen(tmp_path.c_str(), "rb");
        if (!fp) {
            std::filesystem::remove(tmp_path);
            cleanup.untrack_file(tmp_path);
            throw std::runtime_error("Failed to open temp file");
        }
    }

    ~FuzzFile() {
        if (fp) {
            fclose(fp);
        }
        std::filesystem::remove(tmp_path);
        cleanup.untrack_file(tmp_path);
    }

    const char* get_path() { return tmp_path.c_str(); }

    static std::unique_ptr<FuzzFile> from_bytes(const uint8_t* data, size_t size) {
        return std::make_unique<FuzzFile>(data, size);
    }

    FILE* get() { return fp; }
    bool seek(size_t offset, int whence) { return fseek(fp, offset, whence) == 0; }
    void read_raw(void* dst, size_t size) {
        if (fread(dst, 1, size, fp) != size) {
            throw std::runtime_error("Failed to read from file");
        }
    }

private:
    FILE* fp;
    std::filesystem::path tmp_path;
};

void fuzz_chat_template(const uint8_t* data, size_t size) {
    if (size < 10) return;

    // Create chat messages from fuzzer data
    std::vector<llama_chat_message> messages;
    size_t pos = 0;

    // Allocate strings to maintain lifetime
    std::vector<std::string> content_storage;

    while (pos + 4 < size) {
        uint32_t msg_len = *(uint32_t*)(data + pos);
        pos += 4;
        if (pos + msg_len > size) break;

        // Store content string
        content_storage.emplace_back(std::string((char*)(data + pos), msg_len));

        llama_chat_message msg;
        msg.role = (msg_len % 3 == 0) ? "user" :
                   (msg_len % 3 == 1) ? "assistant" : "system";
        msg.content = content_storage.back().c_str(); // Use c_str() to get const char*
        messages.push_back(msg);
        pos += msg_len;
    }

    std::vector<const llama_chat_message*> msg_ptrs;
    for (const auto& msg : messages) {
        msg_ptrs.push_back(&msg);
    }

    std::string result;
    for (llm_chat_template tmpl = LLM_CHAT_TEMPLATE_UNKNOWN;
         tmpl <= LLM_CHAT_TEMPLATE_MEGREZ;
         tmpl = (llm_chat_template)(tmpl + 1)) {
        llm_chat_apply_template(tmpl, msg_ptrs, result, true);
    }
}

void fuzz_tensor_validation(const uint8_t* data, size_t size) {
    if (size < sizeof(int64_t) * 4) return;

    std::vector<int64_t> dims;
    for (size_t i = 0; i < 4 && i * sizeof(int64_t) < size; i++) {
        dims.push_back(*(int64_t*)(data + i * sizeof(int64_t)));
    }

    try {
        auto file = FuzzFile::from_bytes(data, size);
        std::vector<std::string> splits; // Create vector first
        llama_model_loader loader(file->get_path(), splits, false, true, nullptr); // Pass vector by reference

        const char* tensor_names[] = {
            "token_embd.weight",
            "output_norm.weight",
            "output.weight"
        };

        for (const auto& name : tensor_names) {
            loader.check_tensor_dims(name, dims, false);
        }
    } catch (...) {}
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 32) return 0;
    if (size > 100000) return 0;

    try {
        // Existing GGUF format fuzzing
        std::vector<std::string> splits;
        const bool use_mmap = rand() % 2 == 1;
        const bool check_tensors = rand() % 2 == 1;

        auto file = FuzzFile::from_bytes(data, size);
        llama_model_loader loader(file->get_path(), splits, use_mmap, check_tensors, nullptr);

        // Add chat template fuzzing
        fuzz_chat_template(data, size);

        // Add tensor validation fuzzing
        fuzz_tensor_validation(data, size);

        // Fuzz model metadata
        if (size > 8) {
            std::string key(data, data + 8);
            std::string result;
            loader.get_key(key, result, false);
        }

    } catch (...) {
        return 0;
    }

    return 0;
}
