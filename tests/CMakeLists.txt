llama_add_compile_flags()

add_executable(fuzz-gguf fuzz-gguf.cpp)
target_link_libraries(fuzz-gguf PRIVATE common)
target_compile_options(fuzz-gguf PRIVATE -fsanitize=fuzzer -g -O3 -flto=full)
target_link_options(fuzz-gguf PRIVATE -fsanitize=fuzzer -g -O3 -flto=full)
