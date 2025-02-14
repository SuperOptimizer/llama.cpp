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

class GGUFMutator {
public:
   static std::vector<uint8_t> mutate(const uint8_t* data, size_t size, unsigned int seed) {
       std::mt19937 rng(seed);
       
       if (size < 32) return gen_minimal(rng);
       
       std::vector<uint8_t> result(data, data + size);
       
       // Pick mutation strategy
       std::uniform_int_distribution<> strategy(0, 3);
       switch(strategy(rng)) {
           case 0: mutate_header(result, rng); break;
           case 1: mutate_metadata(result, rng); break;
           case 2: mutate_tensors(result, rng); break;
           case 3: mutate_data(result, rng); break;
       }
       
       return result;
   }

private:
   static std::vector<uint8_t> gen_minimal(std::mt19937& rng) {
       std::vector<uint8_t> result;
       result.resize(64);
       
       // Magic + version
       memcpy(result.data(), "GGUF", 4);
       uint32_t ver = 3;
       memcpy(result.data()+4, &ver, 4);
       
       // Empty counts
       uint64_t zero = 0;
       memcpy(result.data()+8, &zero, 8);  // tensor count
       memcpy(result.data()+16, &zero, 8); // metadata count
       
       return result;
   }

   static void mutate_header(std::vector<uint8_t>& data, std::mt19937& rng) {
       if (data.size() < 24) return;
       std::uniform_int_distribution<> val(0, 255);
       data[8+val(rng)%16] = val(rng); // Mutate count bytes
   }

   static void mutate_metadata(std::vector<uint8_t>& data, std::mt19937& rng) {
       if (data.size() < 64) return;
       size_t pos = 24;
       size_t kv_pairs = read_val<uint64_t>(data.data(), pos);
       if (kv_pairs > 32) return;
       
       std::uniform_int_distribution<> val(0, 255);
       // Mutate a random KV pair
       while (pos < data.size() - 32) {
           size_t key_len = read_val<uint64_t>(data.data(), pos);
           if (key_len > 256) break;
           pos += key_len;
           data[pos + val(rng)%8] = val(rng);
           break;
       }
   }

   static void mutate_tensors(std::vector<uint8_t>& data, std::mt19937& rng) {
       if (data.size() < 128) return;
       std::uniform_int_distribution<> val(0, 255);
       size_t pos = data.size()/2 + val(rng)%(data.size()/4);
       data[pos] = val(rng);
   }

   static void mutate_data(std::vector<uint8_t>& data, std::mt19937& rng) {
       if (data.size() < 256) return;
       std::uniform_int_distribution<> val(0, 255);
       size_t pos = data.size() - val(rng)%256;
       data[pos] = val(rng);
   }
};

extern "C" size_t LLVMFuzzerCustomMutator(uint8_t* data, size_t size, size_t max_size, unsigned int seed) {
   auto mutated = GGUFMutator::mutate(data, size, seed);
   if (mutated.size() > max_size) return size;
   memcpy(data, mutated.data(), mutated.size());
   return mutated.size();
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
