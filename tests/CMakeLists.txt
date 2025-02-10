llama_add_compile_flags()

add_executable(fuzz-gguf fuzz-gguf.cpp)
target_link_libraries(fuzz-gguf PRIVATE common)
target_compile_options(fuzz-gguf PRIVATE -fsanitize=fuzzer,address -g -O1)
target_link_options(fuzz-gguf PRIVATE -fsanitize=fuzzer,address)
