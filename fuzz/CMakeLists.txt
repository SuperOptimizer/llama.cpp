# Fuzz targets for llama.cpp
#
# Build with: cmake -DLLAMA_BUILD_FUZZ=ON ..

find_package(Threads REQUIRED)

# Common settings for fuzz targets
set(FUZZ_COMPILE_FLAGS "")
set(FUZZ_LINK_FLAGS "")

# fuzz-llama-gguf: GGUF model loading fuzzer (vocab-only)
set(FUZZ_TARGET fuzz-llama-gguf)
add_executable(${FUZZ_TARGET} ${FUZZ_TARGET}.cpp)
target_link_libraries(${FUZZ_TARGET} PRIVATE llama ggml Threads::Threads)
target_include_directories(${FUZZ_TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/src
)

# fuzz-llama-inference: Full model loading + inference fuzzer
set(FUZZ_TARGET fuzz-llama-inference)
add_executable(${FUZZ_TARGET} ${FUZZ_TARGET}.cpp)
target_link_libraries(${FUZZ_TARGET} PRIVATE llama ggml Threads::Threads)
target_include_directories(${FUZZ_TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/src
)

# fuzz-llama-grammar: GBNF grammar parser fuzzer
set(FUZZ_TARGET fuzz-llama-grammar)
add_executable(${FUZZ_TARGET} ${FUZZ_TARGET}.cpp)
target_link_libraries(${FUZZ_TARGET} PRIVATE llama ggml Threads::Threads)
target_include_directories(${FUZZ_TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/src
)

# fuzz-llama-tokenizer: Tokenization/detokenization fuzzer
set(FUZZ_TARGET fuzz-llama-tokenizer)
add_executable(${FUZZ_TARGET} ${FUZZ_TARGET}.cpp)
target_link_libraries(${FUZZ_TARGET} PRIVATE llama ggml Threads::Threads)
target_include_directories(${FUZZ_TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/src
)

# fuzz-llama-sampling: Sampler chain fuzzer
set(FUZZ_TARGET fuzz-llama-sampling)
add_executable(${FUZZ_TARGET} ${FUZZ_TARGET}.cpp)
target_link_libraries(${FUZZ_TARGET} PRIVATE llama ggml Threads::Threads)
target_include_directories(${FUZZ_TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/src
)

# fuzz-llama-kv-cache: KV cache manipulation fuzzer
set(FUZZ_TARGET fuzz-llama-kv-cache)
add_executable(${FUZZ_TARGET} ${FUZZ_TARGET}.cpp)
target_link_libraries(${FUZZ_TARGET} PRIVATE llama ggml Threads::Threads)
target_include_directories(${FUZZ_TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/src
)

# fuzz-llama-chat: Chat template fuzzer
set(FUZZ_TARGET fuzz-llama-chat)
add_executable(${FUZZ_TARGET} ${FUZZ_TARGET}.cpp)
target_link_libraries(${FUZZ_TARGET} PRIVATE llama ggml Threads::Threads)
target_include_directories(${FUZZ_TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/src
)

# fuzz-llama-lora: LoRA adapter loading fuzzer
set(FUZZ_TARGET fuzz-llama-lora)
add_executable(${FUZZ_TARGET} ${FUZZ_TARGET}.cpp)
target_link_libraries(${FUZZ_TARGET} PRIVATE llama ggml Threads::Threads)
target_include_directories(${FUZZ_TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/src
)

# fuzz-json-schema-to-grammar: JSON Schema to GBNF grammar fuzzer
set(FUZZ_TARGET fuzz-json-schema-to-grammar)
add_executable(${FUZZ_TARGET} ${FUZZ_TARGET}.cpp)
target_link_libraries(${FUZZ_TARGET} PRIVATE common llama ggml Threads::Threads)
target_include_directories(${FUZZ_TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/src
)

# fuzz-server-json: Server JSON request parsing fuzzer
set(FUZZ_TARGET fuzz-server-json)
add_executable(${FUZZ_TARGET} ${FUZZ_TARGET}.cpp)
target_link_libraries(${FUZZ_TARGET} PRIVATE common llama ggml Threads::Threads)
target_include_directories(${FUZZ_TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/src
)

# fuzz-chat-parser: Chat output and tool call parser fuzzer
set(FUZZ_TARGET fuzz-chat-parser)
add_executable(${FUZZ_TARGET} ${FUZZ_TARGET}.cpp)
target_link_libraries(${FUZZ_TARGET} PRIVATE common llama ggml Threads::Threads)
target_include_directories(${FUZZ_TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/src
)

# fuzz-minja-template: Minja Jinja-style template parser fuzzer
set(FUZZ_TARGET fuzz-minja-template)
add_executable(${FUZZ_TARGET} ${FUZZ_TARGET}.cpp)
target_link_libraries(${FUZZ_TARGET} PRIVATE common llama ggml Threads::Threads)
target_include_directories(${FUZZ_TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/vendor
)

# fuzz-common-chat: Common chat functionality fuzzer (tool calls, OpenAI compat)
set(FUZZ_TARGET fuzz-common-chat)
add_executable(${FUZZ_TARGET} ${FUZZ_TARGET}.cpp)
target_link_libraries(${FUZZ_TARGET} PRIVATE common llama ggml Threads::Threads)
target_include_directories(${FUZZ_TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/src
)

# Install fuzz targets to bin directory
install(TARGETS
    fuzz-llama-gguf
    fuzz-llama-inference
    fuzz-llama-grammar
    fuzz-llama-tokenizer
    fuzz-llama-sampling
    fuzz-llama-kv-cache
    fuzz-llama-chat
    fuzz-llama-lora
    fuzz-json-schema-to-grammar
    fuzz-server-json
    fuzz-chat-parser
    fuzz-minja-template
    fuzz-common-chat
    RUNTIME DESTINATION bin
)
