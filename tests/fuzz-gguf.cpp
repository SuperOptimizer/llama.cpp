#include "llama-model-loader.h"
#include "ggml.h"
#include "gguf.h"
#include "llama-chat.h"
#include <cstdint>
#include <cstring>
#include <memory>
#include <filesystem>
#include <random>
#include <stdio.h>
#include <set>

// File cleanup
class GlobalCleanup {
public:
   void track_file(const std::filesystem::path& p) { active_files.insert(p); }
   void untrack_file(const std::filesystem::path& p) { active_files.erase(p); }
   ~GlobalCleanup() {
       for (const auto& p : active_files) std::filesystem::remove(p);
   }
private:
   std::set<std::filesystem::path> active_files;
};

static GlobalCleanup cleanup;

// Temp file handling
class FuzzFile {
public:
   FuzzFile(const uint8_t* data, size_t size) {
       std::string random_str(16, 'x');
       std::random_device rd;
       std::mt19937 gen(rd());
       std::uniform_int_distribution<> dis(0, 35);
       for (int i = 0; i < 16; i++) {
           random_str[i] = dis(gen) < 10 ? '0' + dis(gen) : 'a' + (dis(gen)-10);
       }

       tmp_path = std::filesystem::temp_directory_path() / ("fuzz-" + random_str + ".gguf");
       cleanup.track_file(tmp_path);

       FILE* out = fopen(tmp_path.c_str(), "wb");
       if (!out || fwrite(data, 1, size, out) != size) {
           if (out) fclose(out);
           std::filesystem::remove(tmp_path);
           cleanup.untrack_file(tmp_path);
           throw std::runtime_error("Write failed");
       }
       fclose(out);
   }

   ~FuzzFile() {
       std::filesystem::remove(tmp_path);
       cleanup.untrack_file(tmp_path);
   }

   const char* get_path() const { return tmp_path.c_str(); }

   static std::unique_ptr<FuzzFile> from_bytes(const uint8_t* data, size_t size) {
       return std::make_unique<FuzzFile>(data, size);
   }

private:
   std::filesystem::path tmp_path;
};

template<typename T>
static T read_val(const uint8_t* data, size_t& pos) {
   T val;
   memcpy(&val, data + pos, sizeof(T));
   pos += sizeof(T);
   return val;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
   if (size < 32 || size > 100000) return 0;

   try {
       std::vector<std::string> splits;
       bool use_mmap = rand() % 2;
       bool check_tensors = rand() % 2;

       auto file = FuzzFile::from_bytes(data, size);
       llama_model_loader loader(file->get_path(), splits, use_mmap, check_tensors, nullptr);
   } catch (...) {}

   return 0;
}
