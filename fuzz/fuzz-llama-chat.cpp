/*
 * Chat template fuzzer for llama.cpp
 *
 * Tests the chat template application functionality:
 * - llama_chat_apply_template() - Apply chat template to messages
 * - llama_chat_builtin_templates() - List built-in templates
 *
 * Focus areas:
 * - Malformed Jinja templates
 * - Unicode/UTF-8 in messages
 * - Large message arrays
 * - Edge cases in template rendering
 */

#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

class FuzzInput {
public:
    FuzzInput(const uint8_t* data, size_t size)
        : data_(data), size_(size), pos_(0) {}

    uint8_t u8() {
        return pos_ < size_ ? data_[pos_++] : 0;
    }

    // Get a null-terminated string of up to max_len bytes
    std::string string(size_t max_len = 256) {
        size_t len = u8();
        if (len > max_len) len = max_len;
        if (pos_ + len > size_) len = size_ - pos_;

        std::string s((const char*)(data_ + pos_), len);
        pos_ += len;
        return s;
    }

    // Get remaining data as string
    std::string remaining_string() {
        if (pos_ >= size_) return "";
        std::string s((const char*)(data_ + pos_), size_ - pos_);
        pos_ = size_;
        return s;
    }

    bool has_data() const { return pos_ < size_; }
    size_t remaining() const { return size_ > pos_ ? size_ - pos_ : 0; }

private:
    const uint8_t* data_;
    size_t size_;
    size_t pos_;
};

static void null_log_callback(enum ggml_log_level level, const char* text, void* user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

// Test with built-in templates
static int test_builtin_templates(FuzzInput& in) {
    // Get list of built-in templates
    const char* templates[32];
    int n_templates = llama_chat_builtin_templates(templates, 32);

    if (n_templates <= 0) return 0;

    // Create some chat messages from fuzz input
    int n_messages = in.u8() % 10 + 1;
    std::vector<llama_chat_message> messages;
    std::vector<std::string> roles;
    std::vector<std::string> contents;

    const char* role_options[] = {"system", "user", "assistant", "tool"};

    for (int i = 0; i < n_messages && in.has_data(); i++) {
        uint8_t role_idx = in.u8() % 4;
        roles.push_back(role_options[role_idx]);
        contents.push_back(in.string(512));

        llama_chat_message msg;
        msg.role = roles.back().c_str();
        msg.content = contents.back().c_str();
        messages.push_back(msg);
    }

    if (messages.empty()) return 0;

    // Test with each built-in template - test ALL templates for better coverage
    std::vector<char> output(8192);

    for (int i = 0; i < n_templates; i++) {
        bool add_ass = (i % 2) == 0;  // Alternate add_ass

        int32_t result = llama_chat_apply_template(
            templates[i],
            messages.data(),
            messages.size(),
            add_ass,
            output.data(),
            output.size()
        );

        // If buffer too small, try with larger buffer
        if (result > (int32_t)output.size()) {
            output.resize(result + 1);
            llama_chat_apply_template(
                templates[i],
                messages.data(),
                messages.size(),
                add_ass,
                output.data(),
                output.size()
            );
        }

        // Also test with opposite add_ass value for more coverage
        llama_chat_apply_template(
            templates[i],
            messages.data(),
            messages.size(),
            !add_ass,
            output.data(),
            output.size()
        );
    }

    return 0;
}

// Test with fuzzed template string
static int test_fuzzed_template(FuzzInput& in) {
    // Get template string from input
    std::string tmpl = in.string(2048);
    if (tmpl.empty()) return 0;

    // Create messages
    int n_messages = in.u8() % 5 + 1;
    std::vector<llama_chat_message> messages;
    std::vector<std::string> roles;
    std::vector<std::string> contents;

    const char* role_options[] = {"system", "user", "assistant"};

    for (int i = 0; i < n_messages && in.has_data(); i++) {
        uint8_t role_idx = in.u8() % 3;
        roles.push_back(role_options[role_idx]);
        contents.push_back(in.string(256));

        llama_chat_message msg;
        msg.role = roles.back().c_str();
        msg.content = contents.back().c_str();
        messages.push_back(msg);
    }

    if (messages.empty()) return 0;

    std::vector<char> output(8192);

    // Try with and without add_ass
    for (int add_ass = 0; add_ass <= 1; add_ass++) {
        int32_t result = llama_chat_apply_template(
            tmpl.c_str(),
            messages.data(),
            messages.size(),
            add_ass != 0,
            output.data(),
            output.size()
        );

        if (result > 0 && result <= (int32_t)output.size()) {
            // Template was applied, output is in buffer
            (void)output[0];
        }
    }

    return 0;
}

// Test edge cases
static int test_edge_cases(FuzzInput& in) {
    std::vector<char> output(4096);

    // Test with nullptr/empty template
    llama_chat_message msg = {"user", "Hello"};
    llama_chat_apply_template(
        nullptr,
        &msg,
        1,
        true,
        output.data(),
        output.size()
    );

    // Test with empty messages
    llama_chat_apply_template(
        "chatml",
        nullptr,
        0,
        true,
        output.data(),
        output.size()
    );

    // Test with very small output buffer
    char tiny[4];
    llama_chat_apply_template(
        "chatml",
        &msg,
        1,
        true,
        tiny,
        sizeof(tiny)
    );

    // Test with various role/content combinations
    std::string role = in.string(64);
    std::string content = in.remaining_string();

    llama_chat_message fuzz_msg;
    fuzz_msg.role = role.c_str();
    fuzz_msg.content = content.c_str();

    llama_chat_apply_template(
        "chatml",
        &fuzz_msg,
        1,
        true,
        output.data(),
        output.size()
    );

    return 0;
}

// Test all named templates exhaustively to cover all formatting branches
static int test_all_named_templates(FuzzInput& in) {
    // All known template names from llama-chat.cpp
    const char* template_names[] = {
        "chatml", "llama2", "llama2-sys", "llama2-sys-bos", "llama2-sys-strip",
        "mistral-v1", "mistral-v3", "mistral-v3-tekken", "mistral-v7", "mistral-v7-tekken",
        "phi3", "phi4", "falcon3", "zephyr", "monarch", "gemma", "orion",
        "openchat", "vicuna", "vicuna-orca", "deepseek", "deepseek2", "deepseek3",
        "command-r", "llama3", "chatglm3", "chatglm4", "glmedge", "minicpm",
        "exaone3", "exaone4", "rwkv-world", "granite", "gigachat", "megrez",
        "yandex", "bailing", "bailing-think", "bailing2", "llama4", "smolvlm",
        "hunyuan-moe", "gpt-oss", "hunyuan-dense", "kimi-k2", "seed_oss", "grok-2",
        "pangu-embedded"
    };
    int n_templates = sizeof(template_names) / sizeof(template_names[0]);

    // Create messages with various roles for better coverage
    std::vector<llama_chat_message> messages;
    std::vector<std::string> roles;
    std::vector<std::string> contents;

    // System message
    roles.push_back("system");
    contents.push_back("You are a helpful assistant.");
    messages.push_back({roles.back().c_str(), contents.back().c_str()});

    // User message
    roles.push_back("user");
    contents.push_back(in.string(128));
    if (contents.back().empty()) contents.back() = "Hello";
    messages.push_back({roles.back().c_str(), contents.back().c_str()});

    // Assistant message
    roles.push_back("assistant");
    contents.push_back(in.string(128));
    if (contents.back().empty()) contents.back() = "Hi there!";
    messages.push_back({roles.back().c_str(), contents.back().c_str()});

    // Another user message
    roles.push_back("user");
    contents.push_back("Follow up question");
    messages.push_back({roles.back().c_str(), contents.back().c_str()});

    std::vector<char> output(16384);

    // Test each template with and without add_ass
    for (int i = 0; i < n_templates; i++) {
        for (int add_ass = 0; add_ass <= 1; add_ass++) {
            int32_t result = llama_chat_apply_template(
                template_names[i],
                messages.data(),
                messages.size(),
                add_ass != 0,
                output.data(),
                output.size()
            );
            (void)result;
        }
    }

    // Also test with tool role for templates that support it
    roles.push_back("tool");
    contents.push_back("{\"result\": \"data\"}");
    messages.push_back({roles.back().c_str(), contents.back().c_str()});

    for (int i = 0; i < n_templates; i++) {
        llama_chat_apply_template(
            template_names[i],
            messages.data(),
            messages.size(),
            true,
            output.data(),
            output.size()
        );
    }

    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 8) return 0;
    if (size > 64 * 1024) return 0;

    llama_log_set(null_log_callback, nullptr);
    llama_backend_init();

    FuzzInput in(data, size);
    uint8_t mode = in.u8() % 4;

    switch (mode) {
        case 0:
            test_builtin_templates(in);
            break;
        case 1:
            test_fuzzed_template(in);
            break;
        case 2:
            test_edge_cases(in);
            break;
        case 3:
            test_all_named_templates(in);
            break;
    }

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
