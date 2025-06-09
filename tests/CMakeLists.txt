llama_add_compile_flags()


add_executable(test-fuzz test-fuzz.cpp)
target_link_libraries(test-fuzz PRIVATE common llama)
target_link_options(test-fuzz PRIVATE -fsanitize=fuzzer)
install(TARGETS test-fuzz RUNTIME)