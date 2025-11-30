/*
 * Chat output parser fuzzer for llama.cpp
 *
 * Tests the parsing of LLM output for tool calls and structured content:
 * - common_chat_msg_parser - Main parser class
 * - common_json_parse() - Partial JSON healing
 * - XML tool call parsing
 * - Regex-based content extraction
 *
 * This is a critical attack surface because:
 * - Processes untrusted model output
 * - Complex regex and string parsing
 * - Partial/streaming JSON handling
 * - XML parsing for tool calls
 *
 * Focus areas:
 * - Malformed tool call XML
 * - Partial/incomplete JSON
 * - Edge cases in regex matching
 * - Unicode handling
 * - Deeply nested structures
 */

#include "chat.h"
#include "chat-parser.h"
#include "json-partial.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>

using json = nlohmann::ordered_json;

// Helper class for parsing fuzz input
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

    std::string remaining() {
        if (pos_ >= size_) return "";
        std::string s((const char*)(data_ + pos_), size_ - pos_);
        pos_ = size_;
        return s;
    }

    bool has_data() const { return pos_ < size_; }

private:
    const uint8_t* data_;
    size_t size_;
    size_t pos_;
};

// Test partial JSON parsing and healing
static int test_partial_json(const uint8_t* data, size_t size) {
    std::string input((const char*)data, size);

    // Test with various healing markers
    const char* markers[] = {"", "HEAL", "__MARKER__", "\x00", "\"", "}", "]"};

    for (const char* marker : markers) {
        try {
            common_json result;
            bool ok = common_json_parse(input, marker, result);
            if (ok) {
                // Use result to prevent optimization
                (void)result.json.dump().size();
                (void)result.healing_marker.marker.size();
            }
        } catch (const std::exception&) {
            // Expected for malformed input
        }
    }

    return 0;
}

// Test partial JSON with iterator interface
static int test_partial_json_iterator(const uint8_t* data, size_t size) {
    std::string input((const char*)data, size);

    try {
        auto it = input.cbegin();
        auto end = input.cend();
        common_json result;

        std::string marker = (size > 0) ? std::string(1, (char)data[0]) : "";

        bool ok = common_json_parse(it, end, marker, result);
        if (ok) {
            // Check how far we parsed
            size_t parsed = std::distance(input.cbegin(), it);
            (void)parsed;
            (void)result.json.dump().size();
        }
    } catch (const std::exception&) {
        // Expected
    }

    return 0;
}

// Test common_chat_msg_parser with various inputs
static int test_chat_msg_parser(FuzzInput& in) {
    std::string input = in.remaining();
    if (input.empty()) return 0;

    bool is_partial = in.u8() % 2;

    // Create syntax configuration
    common_chat_syntax syntax;
    syntax.format = static_cast<common_chat_format>(in.u8() % 20); // Various formats

    try {
        common_chat_msg_parser parser(input, is_partial, syntax);

        // Try various parsing operations
        uint8_t ops = in.u8();

        if (ops & 0x01) {
            parser.consume_spaces();
        }

        if (ops & 0x02) {
            try {
                parser.consume_rest();
            } catch (...) {}
        }

        if (ops & 0x04) {
            try {
                auto json_result = parser.try_consume_json();
                if (json_result) {
                    (void)json_result->json.dump().size();
                }
            } catch (...) {}
        }

        if (ops & 0x08) {
            try {
                parser.try_parse_reasoning("<think>", "</think>");
            } catch (...) {}
        }

        // Get the result
        const auto& result = parser.result();
        (void)result.content.size();
        (void)result.tool_calls.size();

    } catch (const common_chat_msg_partial_exception&) {
        // Expected for partial input
    } catch (const std::exception&) {
        // Expected for malformed input
    }

    return 0;
}

// Test XML tool call parsing formats
static int test_xml_tool_call_format(FuzzInput& in) {
    // Create various XML formats to test
    xml_tool_call_format formats[] = {
        // MiniMax-M2 style
        {
            .scope_start = "<minimax:tool_call>\n",
            .tool_start = "<invoke name=\"",
            .tool_sep = "\">\n",
            .key_start = "<parameter name=\"",
            .key_val_sep = "\">",
            .val_end = "</parameter>\n",
            .tool_end = "</invoke>\n",
            .scope_end = "</minimax:tool_call>",
        },
        // GLM 4.5 style
        {
            .scope_start = "",
            .tool_start = "<tool_call>",
            .tool_sep = "\n",
            .key_start = "<arg_key>",
            .key_val_sep = "</arg_key>\n<arg_value>",
            .val_end = "</arg_value>\n",
            .tool_end = "</tool_call>\n",
            .scope_end = "",
        },
        // Simple style
        {
            .scope_start = "",
            .tool_start = "<function=",
            .tool_sep = ">",
            .key_start = "",
            .key_val_sep = "=",
            .val_end = ",",
            .tool_end = "</function>",
            .scope_end = "",
        },
    };

    std::string input = in.remaining();
    if (input.empty()) return 0;

    bool is_partial = in.u8() % 2;
    int format_idx = in.u8() % 3;

    common_chat_syntax syntax;
    syntax.format = COMMON_CHAT_FORMAT_CONTENT_ONLY;

    try {
        common_chat_msg_parser parser(input, is_partial, syntax);

        // Try parsing with the selected format
        bool ok = parser.try_consume_xml_tool_calls(formats[format_idx]);
        (void)ok;

        const auto& result = parser.result();
        (void)result.tool_calls.size();

    } catch (const std::exception&) {
        // Expected
    }

    return 0;
}

// Test reasoning + XML tool call parsing
static int test_reasoning_xml(FuzzInput& in) {
    std::string input = in.remaining();
    if (input.empty()) return 0;

    xml_tool_call_format format = {
        .scope_start = "",
        .tool_start = "<tool_call>",
        .tool_sep = "\n",
        .key_start = "<arg>",
        .key_val_sep = "=",
        .val_end = "</arg>",
        .tool_end = "</tool_call>",
        .scope_end = "",
    };

    bool is_partial = in.u8() % 2;

    common_chat_syntax syntax;
    syntax.format = COMMON_CHAT_FORMAT_CONTENT_ONLY;

    try {
        common_chat_msg_parser parser(input, is_partial, syntax);

        // Test reasoning with various think tags
        const char* think_starts[] = {"<think>", "<reasoning>", "<<thought>>", "<|think|>"};
        const char* think_ends[] = {"</think>", "</reasoning>", "<</thought>>", "<|/think|>"};

        int tag_idx = in.u8() % 4;
        parser.consume_reasoning_with_xml_tool_calls(format, think_starts[tag_idx], think_ends[tag_idx]);

        const auto& result = parser.result();
        (void)result.reasoning_content.size();
        (void)result.content.size();
        (void)result.tool_calls.size();

    } catch (const std::exception&) {
        // Expected
    }

    return 0;
}

// Test tool call addition
static int test_tool_call_add(FuzzInput& in) {
    std::string input = in.string(64);
    if (input.empty()) input = "test";

    common_chat_syntax syntax;
    syntax.format = COMMON_CHAT_FORMAT_CONTENT_ONLY;

    try {
        common_chat_msg_parser parser(input, false, syntax);

        // Test add_tool_call with strings
        std::string name = in.string(64);
        std::string id = in.string(32);
        std::string arguments = in.string(256);
        parser.add_tool_call(name, id, arguments);

        // Test add_tool_call with JSON
        json tool_call;
        tool_call["name"] = in.string(64);
        tool_call["id"] = in.string(32);

        if (in.u8() % 2) {
            tool_call["arguments"] = in.string(256);
        } else {
            json args;
            args["key"] = in.string(32);
            tool_call["arguments"] = args;
        }
        parser.add_tool_call(tool_call);

        // Test add_tool_call_short_form
        json short_form;
        std::string func_name = in.string(32);
        if (func_name.empty()) func_name = "test_func";
        json func_args;
        func_args["param1"] = in.string(32);
        short_form[func_name] = func_args;
        parser.add_tool_call_short_form(short_form);

        // Test add_tool_calls array
        json calls = json::array();
        calls.push_back(tool_call);
        parser.add_tool_calls(calls);

        const auto& result = parser.result();
        (void)result.tool_calls.size();

    } catch (const std::exception&) {
        // Expected
    }

    return 0;
}

// Test literal and regex consumption
static int test_consume_patterns(FuzzInput& in) {
    std::string input = in.remaining();
    if (input.size() < 10) return 0;

    common_chat_syntax syntax;
    syntax.format = COMMON_CHAT_FORMAT_CONTENT_ONLY;

    try {
        common_chat_msg_parser parser(input, false, syntax);

        // Try consuming various literals
        std::string literal = in.string(32);
        parser.try_consume_literal(literal);

        // Try finding literals
        std::string search = in.string(16);
        auto find_result = parser.try_find_literal(search);
        if (find_result) {
            (void)find_result->prelude.size();
        }

        // Content/reasoning addition
        parser.add_content(in.string(64));
        parser.add_reasoning_content(in.string(64));

        // Position manipulation
        size_t pos = parser.pos();
        if (pos > 5) {
            parser.move_back(5);
        }
        if (input.size() > 10) {
            parser.move_to(input.size() / 2);
        }

        parser.finish();

        const auto& result = parser.result();
        (void)result.content.size();

    } catch (const std::exception&) {
        // Expected
    }

    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 2) return 0;
    if (size > 64 * 1024) return 0;

    uint8_t mode = data[0] % 8;
    data++;
    size--;

    FuzzInput in(data, size);

    switch (mode) {
        case 0:
            test_partial_json(data, size);
            break;
        case 1:
            test_partial_json_iterator(data, size);
            break;
        case 2:
            test_chat_msg_parser(in);
            break;
        case 3:
            test_xml_tool_call_format(in);
            break;
        case 4:
            test_reasoning_xml(in);
            break;
        case 5:
            test_tool_call_add(in);
            break;
        case 6:
            test_consume_patterns(in);
            break;
        case 7:
            // Raw string test for partial JSON
            {
                std::string raw((const char*)data, size);
                common_json result;
                try {
                    common_json_parse(raw, "", result);
                } catch (...) {}
            }
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
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (fsize <= 0 || fsize > 100 * 1024 * 1024) {
        fclose(f);
        return 1;
    }

    uint8_t* data = (uint8_t*)malloc(fsize);
    if (!data) {
        fclose(f);
        return 1;
    }

    if (fread(data, 1, fsize, f) != (size_t)fsize) {
        free(data);
        fclose(f);
        return 1;
    }
    fclose(f);

    int result = LLVMFuzzerTestOneInput(data, fsize);

    free(data);
    return result;
}
#endif
