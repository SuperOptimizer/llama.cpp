/*
 * Common chat functionality fuzzer for llama.cpp
 *
 * Tests the higher-level chat functions in common/chat.cpp:
 * - common_chat_parse() - Parse chat output with tool calls
 * - common_chat_msgs_parse_oaicompat() - Parse OpenAI-format messages
 * - common_chat_tools_parse_oaicompat() - Parse OpenAI-format tools
 * - common_chat_verify_template() - Verify chat templates
 * - common_chat_tool_choice_parse_oaicompat() - Parse tool choice
 * - common_chat_msg_diff::compute_diffs() - Compute message diffs
 *
 * These functions had ~2% coverage before this harness.
 */

#include "chat.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

class FuzzInput {
public:
    FuzzInput(const uint8_t* data, size_t size)
        : data_(data), size_(size), pos_(0) {}

    uint8_t u8() {
        return pos_ < size_ ? data_[pos_++] : 0;
    }

    std::string string(size_t max_len = 256) {
        size_t len = u8();
        if (len > max_len) len = max_len;
        if (pos_ + len > size_) len = size_ - pos_;

        std::string s((const char*)(data_ + pos_), len);
        pos_ += len;
        return s;
    }

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

// Test common_chat_parse() with various chat formats
static void test_chat_parse(FuzzInput& in) {
    std::string input = in.remaining_string();
    if (input.empty()) return;

    // Test with different chat formats
    common_chat_format formats[] = {
        COMMON_CHAT_FORMAT_CONTENT_ONLY,
        COMMON_CHAT_FORMAT_GENERIC,
        COMMON_CHAT_FORMAT_MISTRAL_NEMO,
        COMMON_CHAT_FORMAT_LLAMA_3_X,
        COMMON_CHAT_FORMAT_DEEPSEEK_R1,
        COMMON_CHAT_FORMAT_HERMES_2_PRO,
        COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2,
    };

    for (auto format : formats) {
        common_chat_syntax syntax;
        syntax.format = format;
        syntax.reasoning_format = COMMON_REASONING_FORMAT_NONE;
        syntax.parse_tool_calls = true;

        try {
            // Test partial parsing
            common_chat_msg msg = common_chat_parse(input, true, syntax);
            (void)msg.content;
            (void)msg.tool_calls.size();
            (void)msg.reasoning_content;

            // Test complete parsing
            msg = common_chat_parse(input, false, syntax);
            (void)msg.content;
        } catch (...) {
            // Expected for malformed input
        }
    }

    // Test with reasoning formats
    common_reasoning_format reasoning_formats[] = {
        COMMON_REASONING_FORMAT_NONE,
        COMMON_REASONING_FORMAT_AUTO,
        COMMON_REASONING_FORMAT_DEEPSEEK,
        COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY,
    };

    for (auto rf : reasoning_formats) {
        common_chat_syntax syntax;
        syntax.format = COMMON_CHAT_FORMAT_GENERIC;
        syntax.reasoning_format = rf;
        syntax.reasoning_in_content = (in.u8() & 1) != 0;

        try {
            common_chat_msg msg = common_chat_parse(input, false, syntax);
            (void)msg.reasoning_content;
        } catch (...) {
            // Expected
        }
    }
}

// Test OpenAI-compatible message parsing
static void test_msgs_parse_oaicompat(FuzzInput& in) {
    std::string json_str = in.remaining_string();
    if (json_str.empty()) return;

    try {
        std::vector<common_chat_msg> msgs = common_chat_msgs_parse_oaicompat(json_str);
        for (const auto& msg : msgs) {
            (void)msg.role;
            (void)msg.content;
            (void)msg.tool_calls.size();
        }
    } catch (...) {
        // Expected for invalid JSON
    }
}

// Test OpenAI-compatible tools parsing
static void test_tools_parse_oaicompat(FuzzInput& in) {
    std::string json_str = in.remaining_string();
    if (json_str.empty()) return;

    try {
        std::vector<common_chat_tool> tools = common_chat_tools_parse_oaicompat(json_str);
        for (const auto& tool : tools) {
            (void)tool.name;
            (void)tool.description;
            (void)tool.parameters;
        }
    } catch (...) {
        // Expected for invalid JSON
    }
}

// Test template verification
static void test_verify_template(FuzzInput& in) {
    std::string tmpl = in.remaining_string();
    if (tmpl.empty()) return;

    // Test without Jinja
    bool valid_no_jinja = common_chat_verify_template(tmpl, false);
    (void)valid_no_jinja;

    // Test with Jinja
    try {
        bool valid_jinja = common_chat_verify_template(tmpl, true);
        (void)valid_jinja;
    } catch (...) {
        // May throw for malformed templates
    }
}

// Test tool choice parsing
static void test_tool_choice_parse(FuzzInput& in) {
    std::string choice = in.string(64);
    if (choice.empty()) return;

    common_chat_tool_choice result = common_chat_tool_choice_parse_oaicompat(choice);
    (void)result;

    // Test known values
    (void)common_chat_tool_choice_parse_oaicompat("auto");
    (void)common_chat_tool_choice_parse_oaicompat("none");
    (void)common_chat_tool_choice_parse_oaicompat("required");
}

// Test message diff computation
static void test_msg_diff(FuzzInput& in) {
    common_chat_msg prev_msg;
    prev_msg.role = "assistant";
    prev_msg.content = in.string(256);
    prev_msg.reasoning_content = in.string(128);

    // Add some tool calls to previous
    int n_prev_tools = in.u8() % 3;
    for (int i = 0; i < n_prev_tools; i++) {
        common_chat_tool_call tc;
        tc.name = in.string(32);
        tc.arguments = in.string(128);
        tc.id = in.string(16);
        prev_msg.tool_calls.push_back(tc);
    }

    common_chat_msg new_msg;
    new_msg.role = "assistant";
    new_msg.content = prev_msg.content + in.string(128);
    new_msg.reasoning_content = prev_msg.reasoning_content + in.string(64);

    // Copy existing tool calls and maybe add more
    new_msg.tool_calls = prev_msg.tool_calls;
    if (!new_msg.tool_calls.empty()) {
        // Extend arguments of last tool
        new_msg.tool_calls.back().arguments += in.string(64);
    }

    int n_new_tools = in.u8() % 2;
    for (int i = 0; i < n_new_tools; i++) {
        common_chat_tool_call tc;
        tc.name = in.string(32);
        tc.arguments = in.string(128);
        tc.id = in.string(16);
        new_msg.tool_calls.push_back(tc);
    }

    try {
        auto diffs = common_chat_msg_diff::compute_diffs(prev_msg, new_msg);
        for (const auto& diff : diffs) {
            (void)diff.content_delta;
            (void)diff.reasoning_content_delta;
            (void)diff.tool_call_index;
        }
    } catch (...) {
        // May throw for invalid diffs
    }
}

// Test format name functions
static void test_format_names(FuzzInput& in) {
    // Test all chat format names
    for (int i = 0; i < COMMON_CHAT_FORMAT_COUNT; i++) {
        const char* name = common_chat_format_name(static_cast<common_chat_format>(i));
        (void)name;
    }

    // Test reasoning format names
    common_reasoning_format rf_values[] = {
        COMMON_REASONING_FORMAT_NONE,
        COMMON_REASONING_FORMAT_AUTO,
        COMMON_REASONING_FORMAT_DEEPSEEK,
        COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY,
    };
    for (auto rf : rf_values) {
        const char* name = common_reasoning_format_name(rf);
        (void)name;
    }

    // Test parsing reasoning format from name
    std::string format_name = in.string(32);
    common_reasoning_format rf = common_reasoning_format_from_name(format_name);
    (void)rf;

    // Test known format names
    (void)common_reasoning_format_from_name("none");
    (void)common_reasoning_format_from_name("deepseek");
    (void)common_reasoning_format_from_name("thinking");
}

// Test message construction (builds message structures like what the JSON conversion uses)
static void test_msg_construction(FuzzInput& in) {
    common_chat_msg msg;
    msg.role = "assistant";
    msg.content = in.string(256);
    msg.reasoning_content = in.string(128);

    int n_tools = in.u8() % 3;
    for (int i = 0; i < n_tools; i++) {
        common_chat_tool_call tc;
        tc.name = in.string(32);
        tc.arguments = in.string(128);
        tc.id = in.string(16);
        msg.tool_calls.push_back(tc);
    }

    // Access fields to prevent optimization
    (void)msg.role.size();
    (void)msg.content.size();
    (void)msg.reasoning_content.size();
    (void)msg.tool_calls.size();
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 4) return 0;
    if (size > 64 * 1024) return 0;

    llama_log_set(null_log_callback, nullptr);

    FuzzInput in(data, size);
    uint8_t mode = in.u8() % 8;

    switch (mode) {
        case 0:
            test_chat_parse(in);
            break;
        case 1:
            test_msgs_parse_oaicompat(in);
            break;
        case 2:
            test_tools_parse_oaicompat(in);
            break;
        case 3:
            test_verify_template(in);
            break;
        case 4:
            test_tool_choice_parse(in);
            break;
        case 5:
            test_msg_diff(in);
            break;
        case 6:
            test_format_names(in);
            break;
        case 7:
            test_msg_construction(in);
            break;
    }

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
