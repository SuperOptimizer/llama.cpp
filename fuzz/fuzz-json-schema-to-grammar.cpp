/*
 * JSON Schema to Grammar fuzzer for llama.cpp
 *
 * Tests the json_schema_to_grammar() function which converts
 * OpenAI-style JSON schemas to GBNF grammars.
 *
 * This is a high-value target because:
 * - User-controlled input via server API (response_format.json_schema)
 * - Complex recursive parsing with regex
 * - String manipulation and grammar generation
 *
 * Focus areas:
 * - Malformed JSON schemas
 * - Deeply nested object/array schemas
 * - Edge cases in integer range constraints
 * - String pattern constraints
 * - $ref and $defs resolution
 * - Enum types and const values
 */

#include "json-schema-to-grammar.h"
#include "common.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <stdexcept>

using json = nlohmann::ordered_json;

// Suppress logging during fuzzing
static void null_log_callback(enum ggml_log_level level, const char* text, void* user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

// Test with raw JSON schema string
static int test_json_schema(const uint8_t* data, size_t size) {
    // Create null-terminated string
    std::string schema_str((const char*)data, size);

    try {
        // Parse as JSON first
        json schema = json::parse(schema_str);

        // Convert to grammar
        std::string grammar = json_schema_to_grammar(schema, false);

        // Also test with force_gbnf=true
        std::string grammar_gbnf = json_schema_to_grammar(schema, true);

        // Use the results to prevent optimization
        (void)grammar.size();
        (void)grammar_gbnf.size();

    } catch (const json::parse_error&) {
        // Invalid JSON - expected
    } catch (const std::exception&) {
        // Other errors during conversion - expected for malformed schemas
    }

    return 0;
}

// Test with structured schema generation
static int test_structured_schema(const uint8_t* data, size_t size) {
    if (size < 4) return 0;

    size_t pos = 0;
    auto read_u8 = [&]() -> uint8_t {
        return pos < size ? data[pos++] : 0;
    };
    auto read_string = [&](size_t max_len) -> std::string {
        size_t len = read_u8();
        if (len > max_len) len = max_len;
        if (pos + len > size) len = size - pos;
        std::string s((const char*)(data + pos), len);
        pos += len;
        return s;
    };

    try {
        json schema;

        uint8_t type_selector = read_u8() % 10;

        switch (type_selector) {
            case 0: // string type with pattern
                schema["type"] = "string";
                if (read_u8() % 2) {
                    schema["pattern"] = read_string(128);
                }
                if (read_u8() % 2) {
                    schema["minLength"] = read_u8();
                    schema["maxLength"] = read_u8() + schema["minLength"].get<int>();
                }
                if (read_u8() % 2) {
                    schema["format"] = read_string(32);
                }
                break;

            case 1: // integer type with range
                schema["type"] = "integer";
                if (read_u8() % 2) {
                    int64_t min_val = (int64_t)read_u8() - 128;
                    schema["minimum"] = min_val;
                }
                if (read_u8() % 2) {
                    int64_t max_val = (int64_t)read_u8() + 128;
                    schema["maximum"] = max_val;
                }
                if (read_u8() % 2) {
                    schema["exclusiveMinimum"] = (read_u8() % 2) != 0;
                }
                break;

            case 2: // number type
                schema["type"] = "number";
                if (read_u8() % 2) {
                    schema["minimum"] = (double)read_u8() / 10.0;
                    schema["maximum"] = (double)read_u8() + 10.0;
                }
                break;

            case 3: // boolean type
                schema["type"] = "boolean";
                break;

            case 4: // null type
                schema["type"] = "null";
                break;

            case 5: // array type
                schema["type"] = "array";
                {
                    json items;
                    items["type"] = (read_u8() % 2) ? "string" : "integer";
                    schema["items"] = items;
                }
                if (read_u8() % 2) {
                    schema["minItems"] = read_u8() % 10;
                    schema["maxItems"] = read_u8() % 20 + 1;
                }
                break;

            case 6: // object type with properties
                schema["type"] = "object";
                {
                    json properties;
                    int n_props = read_u8() % 5 + 1;
                    json required = json::array();

                    for (int i = 0; i < n_props && pos < size; i++) {
                        std::string prop_name = read_string(32);
                        if (prop_name.empty()) prop_name = "prop" + std::to_string(i);

                        json prop;
                        prop["type"] = (read_u8() % 2) ? "string" : "integer";
                        if (read_u8() % 2) {
                            prop["description"] = read_string(64);
                        }
                        properties[prop_name] = prop;

                        if (read_u8() % 2) {
                            required.push_back(prop_name);
                        }
                    }
                    schema["properties"] = properties;
                    if (!required.empty()) {
                        schema["required"] = required;
                    }
                }
                if (read_u8() % 2) {
                    schema["additionalProperties"] = (read_u8() % 2) != 0;
                }
                break;

            case 7: // enum type
                schema["enum"] = json::array();
                {
                    int n_values = read_u8() % 8 + 1;
                    for (int i = 0; i < n_values && pos < size; i++) {
                        if (read_u8() % 2) {
                            schema["enum"].push_back(read_string(32));
                        } else {
                            schema["enum"].push_back((int)read_u8());
                        }
                    }
                }
                break;

            case 8: // const value
                if (read_u8() % 2) {
                    schema["const"] = read_string(64);
                } else {
                    schema["const"] = (int)read_u8();
                }
                break;

            case 9: // anyOf/oneOf
                {
                    std::string combinator = (read_u8() % 2) ? "anyOf" : "oneOf";
                    schema[combinator] = json::array();
                    int n_options = read_u8() % 4 + 1;
                    for (int i = 0; i < n_options; i++) {
                        json option;
                        option["type"] = (read_u8() % 2) ? "string" : "integer";
                        schema[combinator].push_back(option);
                    }
                }
                break;
        }

        // Convert to grammar
        std::string grammar = json_schema_to_grammar(schema, false);
        (void)grammar.size();

    } catch (const std::exception&) {
        // Expected for invalid schemas
    }

    return 0;
}

// Test with nested schemas and $ref
static int test_nested_schema(const uint8_t* data, size_t size) {
    if (size < 8) return 0;

    size_t pos = 0;
    auto read_u8 = [&]() -> uint8_t {
        return pos < size ? data[pos++] : 0;
    };

    try {
        json schema;
        schema["type"] = "object";

        // Add $defs for references
        json defs;
        int n_defs = read_u8() % 3 + 1;
        for (int i = 0; i < n_defs; i++) {
            std::string def_name = "def" + std::to_string(i);
            json def;
            def["type"] = (read_u8() % 2) ? "string" : "integer";
            defs[def_name] = def;
        }
        schema["$defs"] = defs;

        // Properties with $ref
        json properties;
        int n_props = read_u8() % 4 + 1;
        for (int i = 0; i < n_props; i++) {
            std::string prop_name = "prop" + std::to_string(i);
            json prop;
            if (read_u8() % 2) {
                // Reference to a definition
                int ref_idx = read_u8() % n_defs;
                prop["$ref"] = "#/$defs/def" + std::to_string(ref_idx);
            } else {
                prop["type"] = "string";
            }
            properties[prop_name] = prop;
        }
        schema["properties"] = properties;

        std::string grammar = json_schema_to_grammar(schema, false);
        (void)grammar.size();

    } catch (const std::exception&) {
        // Expected
    }

    return 0;
}

// Test gbnf_format_literal
static int test_format_literal(const uint8_t* data, size_t size) {
    std::string literal((const char*)data, size);

    try {
        std::string formatted = gbnf_format_literal(literal);
        (void)formatted.size();
    } catch (const std::exception&) {
        // Expected for invalid literals
    }

    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 2) return 0;
    if (size > 64 * 1024) return 0;

    // Select test mode based on first byte
    uint8_t mode = data[0] % 4;
    data++;
    size--;

    switch (mode) {
        case 0:
            test_json_schema(data, size);
            break;
        case 1:
            test_structured_schema(data, size);
            break;
        case 2:
            test_nested_schema(data, size);
            break;
        case 3:
            test_format_literal(data, size);
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
