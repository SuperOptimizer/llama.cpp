// GGUF format fuzzer for llama.cpp
// Compile with: clang++ -g -O1 -fsanitize=fuzzer,address gguf_fuzzer.cpp -I../llama.cpp

#include "llama-model-loader.h"
#include "ggml.h"

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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 32) {
        return 0;
    }
    if (size > 100000) return 0;

    try {
        std::vector<std::string> splits;
        const bool use_mmap = rand() % 2 == 1;
        const bool check_tensors = rand() % 2 == 1;

        auto file = FuzzFile::from_bytes(data, size);
        llama_model_loader loader(file->get_path(), splits, use_mmap, check_tensors, nullptr);
    } catch (...) {
        return 0;
    }

    return 0;
}
