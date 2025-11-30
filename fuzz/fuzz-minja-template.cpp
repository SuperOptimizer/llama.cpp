/*
 * Minja template engine fuzzer for llama.cpp
 *
 * Tests the Minja Jinja-style template parser and executor:
 * - Template parsing (minja::Parser::parse)
 * - Template rendering with various contexts
 * - chat_template class for chat formatting
 *
 * Focus areas:
 * - Malformed Jinja syntax
 * - Deeply nested expressions
 * - Edge cases in filters and control flow
 * - Unicode handling
 */

#include "minja/minja.hpp"
#include "minja/chat-template.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using json = nlohmann::ordered_json;

class FuzzInput {
public:
    FuzzInput(const uint8_t* data, size_t size)
        : data_(data), size_(size), pos_(0) {}

    uint8_t u8() {
        return pos_ < size_ ? data_[pos_++] : 0;
    }

    uint16_t u16() {
        uint16_t v = 0;
        if (pos_ + 2 <= size_) {
            memcpy(&v, data_ + pos_, 2);
            pos_ += 2;
        }
        return v;
    }

    std::string string(size_t max_len = 256) {
        size_t len = u16() % (max_len + 1);
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

// Test direct template parsing and rendering
static int test_template_parse(FuzzInput& in) {
    std::string template_str = in.remaining_string();
    if (template_str.empty()) return 0;

    try {
        minja::Options opts;
        opts.trim_blocks = (in.u8() % 2) != 0;
        opts.lstrip_blocks = (in.u8() % 2) != 0;
        opts.keep_trailing_newline = (in.u8() % 2) != 0;

        auto tmpl = minja::Parser::parse(template_str, opts);
        if (!tmpl) return 0;

        // Create a simple context and render
        auto context = minja::Context::make(json({
            {"name", "test"},
            {"value", 42},
            {"items", json::array({1, 2, 3})},
            {"flag", true},
        }));

        std::string result = tmpl->render(context);
        (void)result;
    } catch (const std::exception& e) {
        // Expected for malformed templates
        (void)e;
    }

    return 0;
}

// Test template with fuzzed context
static int test_template_context(FuzzInput& in) {
    // Use a known-good template structure but fuzz the context
    const char* templates[] = {
        "{{ name }}",
        "{% if flag %}yes{% else %}no{% endif %}",
        "{% for item in items %}{{ item }}{% endfor %}",
        "{{ value | default('none') }}",
        "{{ items | length }}",
        "{{ name | upper }}",
        "{{ items | first }}",
        "{{ items | last }}",
        "{% if items %}has items{% endif %}",
        "{{ value + 1 }}",
        "{{ name ~ ' suffix' }}",
    };

    uint8_t tmpl_idx = in.u8() % (sizeof(templates) / sizeof(templates[0]));

    try {
        auto tmpl = minja::Parser::parse(templates[tmpl_idx], {});
        if (!tmpl) return 0;

        // Build fuzzed context
        json ctx_json;

        // Add fuzzed string value
        ctx_json["name"] = in.string(64);

        // Add fuzzed number
        ctx_json["value"] = (int32_t)(in.u16() - 32768);

        // Add fuzzed boolean
        ctx_json["flag"] = (in.u8() % 2) != 0;

        // Add fuzzed array
        json arr = json::array();
        int arr_len = in.u8() % 10;
        for (int i = 0; i < arr_len && in.has_data(); i++) {
            uint8_t type = in.u8() % 4;
            switch (type) {
                case 0: arr.push_back(in.string(32)); break;
                case 1: arr.push_back((int)in.u16()); break;
                case 2: arr.push_back((in.u8() % 2) != 0); break;
                case 3: arr.push_back(nullptr); break;
            }
        }
        ctx_json["items"] = arr;

        auto context = minja::Context::make(ctx_json);
        std::string result = tmpl->render(context);
        (void)result;
    } catch (const std::exception& e) {
        (void)e;
    }

    return 0;
}

// Test chat_template class
static int test_chat_template(FuzzInput& in) {
    std::string template_str = in.string(2048);
    if (template_str.empty()) return 0;

    std::string bos = in.string(16);
    std::string eos = in.string(16);

    try {
        minja::chat_template chat_tmpl(template_str, bos, eos);

        // Create messages
        json messages = json::array();
        int n_messages = in.u8() % 5 + 1;

        const char* roles[] = {"system", "user", "assistant", "tool"};

        for (int i = 0; i < n_messages && in.has_data(); i++) {
            json msg;
            msg["role"] = roles[in.u8() % 4];
            msg["content"] = in.string(256);

            // Sometimes add tool_calls for assistant
            if (msg["role"] == "assistant" && (in.u8() % 3) == 0) {
                json tool_calls = json::array();
                tool_calls.push_back({
                    {"id", "call_" + std::to_string(in.u16())},
                    {"type", "function"},
                    {"function", {
                        {"name", in.string(32)},
                        {"arguments", "{\"arg\": \"value\"}"},
                    }},
                });
                msg["tool_calls"] = tool_calls;
            }

            messages.push_back(msg);
        }

        // Create tools (sometimes)
        json tools;
        if (in.u8() % 2) {
            tools = json::array();
            int n_tools = in.u8() % 3 + 1;
            for (int i = 0; i < n_tools; i++) {
                tools.push_back({
                    {"name", in.string(32)},
                    {"type", "function"},
                    {"function", {
                        {"name", in.string(32)},
                        {"description", in.string(64)},
                        {"parameters", {
                            {"type", "object"},
                            {"properties", json::object()},
                        }},
                    }},
                });
            }
        }

        minja::chat_template_inputs inputs;
        inputs.messages = messages;
        inputs.tools = tools;
        inputs.add_generation_prompt = (in.u8() % 2) != 0;

        minja::chat_template_options opts;
        opts.apply_polyfills = (in.u8() % 2) != 0;

        std::string result = chat_tmpl.apply(inputs, opts);
        (void)result;
    } catch (const std::exception& e) {
        (void)e;
    }

    return 0;
}

// Test expression parsing
static int test_expressions(FuzzInput& in) {
    // Templates that stress expression parsing
    std::string expr = in.string(512);
    if (expr.empty()) return 0;

    // Wrap in expression syntax
    std::string templates[] = {
        "{{ " + expr + " }}",
        "{% if " + expr + " %}yes{% endif %}",
        "{% set x = " + expr + " %}{{ x }}",
    };

    for (const auto& tmpl_str : templates) {
        try {
            auto tmpl = minja::Parser::parse(tmpl_str, {});
            if (tmpl) {
                auto context = minja::Context::make(json({
                    {"a", 1}, {"b", 2}, {"c", "str"},
                    {"arr", json::array({1, 2, 3})},
                    {"obj", {{"x", 1}, {"y", 2}}},
                }));
                std::string result = tmpl->render(context);
                (void)result;
            }
        } catch (const std::exception& e) {
            (void)e;
        }
    }

    return 0;
}

// Test complex control flow
static int test_control_flow(FuzzInput& in) {
    std::string body = in.string(1024);

    // Test various control structures with fuzzed bodies
    std::string templates[] = {
        "{% for i in range(3) %}" + body + "{% endfor %}",
        "{% if true %}" + body + "{% elif false %}x{% else %}y{% endif %}",
        "{% macro m() %}" + body + "{% endmacro %}{{ m() }}",
        "{% set ns = namespace(x=0) %}{% for i in range(2) %}{% set ns.x = ns.x + 1 %}" + body + "{% endfor %}{{ ns.x }}",
    };

    for (const auto& tmpl_str : templates) {
        try {
            auto tmpl = minja::Parser::parse(tmpl_str, {});
            if (tmpl) {
                auto context = minja::Context::make(json::object());
                std::string result = tmpl->render(context);
                (void)result;
            }
        } catch (const std::exception& e) {
            (void)e;
        }
    }

    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 8) return 0;
    if (size > 64 * 1024) return 0;

    FuzzInput in(data, size);
    uint8_t mode = in.u8() % 5;

    switch (mode) {
        case 0:
            test_template_parse(in);
            break;
        case 1:
            test_template_context(in);
            break;
        case 2:
            test_chat_template(in);
            break;
        case 3:
            test_expressions(in);
            break;
        case 4:
            test_control_flow(in);
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
