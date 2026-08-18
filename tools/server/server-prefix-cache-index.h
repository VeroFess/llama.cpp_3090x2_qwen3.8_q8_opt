#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class server_prefix_cache_index {
public:
    explicit server_prefix_cache_index(size_t block_size);

    bool insert(uint64_t entry_id, const std::vector<llama_token> & tokens, const std::string & fingerprint);
    bool erase(uint64_t entry_id);
    uint64_t find_longest_prefix(const std::vector<llama_token> & tokens, const std::string & fingerprint, size_t * matched_tokens) const;
    size_t size() const;

private:
    struct terminal {
        uint64_t entry_id = 0;
        std::vector<llama_token> tail;
        size_t total_tokens = 0;
    };

    struct node {
        std::map<std::vector<llama_token>, std::unique_ptr<node>> children;
        std::vector<terminal> terminals;
    };

    struct record {
        std::string fingerprint;
        std::vector<llama_token> tokens;
    };

    size_t block_size;
    std::unordered_map<std::string, std::unique_ptr<node>> roots;
    std::unordered_map<uint64_t, record> records;
};
