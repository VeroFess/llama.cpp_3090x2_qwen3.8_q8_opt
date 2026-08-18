#include "server-prefix-cache-index.h"

#include <algorithm>

server_prefix_cache_index::server_prefix_cache_index(size_t block_size) : block_size(block_size) {}

bool server_prefix_cache_index::insert(uint64_t entry_id, const std::vector<llama_token> & tokens, const std::string & fingerprint) {
    if (entry_id == 0 || block_size == 0 || tokens.empty() || records.find(entry_id) != records.end()) {
        return false;
    }

    auto & root = roots[fingerprint];
    if (!root) {
        root = std::make_unique<node>();
    }

    node * current = root.get();
    size_t offset = 0;
    while (tokens.size() - offset >= block_size) {
        std::vector<llama_token> block(tokens.begin() + offset, tokens.begin() + offset + block_size);
        auto & child = current->children[block];
        if (!child) {
            child = std::make_unique<node>();
        }
        current = child.get();
        offset += block_size;
    }

    current->terminals.push_back({ entry_id, { tokens.begin() + offset, tokens.end() }, tokens.size() });
    records.emplace(entry_id, record { fingerprint, tokens });
    return true;
}

bool server_prefix_cache_index::erase(uint64_t entry_id) {
    const auto record_it = records.find(entry_id);
    if (record_it == records.end()) {
        return false;
    }

    const auto root_it = roots.find(record_it->second.fingerprint);
    if (root_it == roots.end()) {
        return false;
    }

    node * current = root_it->second.get();
    size_t offset = 0;
    const auto & tokens = record_it->second.tokens;
    while (tokens.size() - offset >= block_size) {
        std::vector<llama_token> block(tokens.begin() + offset, tokens.begin() + offset + block_size);
        const auto child = current->children.find(block);
        if (child == current->children.end()) {
            return false;
        }
        current = child->second.get();
        offset += block_size;
    }

    const auto terminal_it = std::find_if(current->terminals.begin(), current->terminals.end(), [&](const terminal & value) {
        return value.entry_id == entry_id;
    });
    if (terminal_it == current->terminals.end()) {
        return false;
    }
    current->terminals.erase(terminal_it);
    records.erase(record_it);
    return true;
}

uint64_t server_prefix_cache_index::find_longest_prefix(
        const std::vector<llama_token> & tokens,
        const std::string & fingerprint,
        size_t * matched_tokens) const {
    if (matched_tokens != nullptr) {
        *matched_tokens = 0;
    }
    const auto root_it = roots.find(fingerprint);
    if (root_it == roots.end()) {
        return 0;
    }

    const node * current = root_it->second.get();
    uint64_t best_id = 0;
    size_t best_tokens = 0;
    size_t offset = 0;
    while (true) {
        for (const auto & value : current->terminals) {
            if (value.tail.size() <= tokens.size() - offset &&
                    std::equal(value.tail.begin(), value.tail.end(), tokens.begin() + offset) &&
                    value.total_tokens > best_tokens) {
                best_id = value.entry_id;
                best_tokens = value.total_tokens;
            }
        }

        if (tokens.size() - offset < block_size) {
            break;
        }
        std::vector<llama_token> block(tokens.begin() + offset, tokens.begin() + offset + block_size);
        const auto child = current->children.find(block);
        if (child == current->children.end()) {
            break;
        }
        current = child->second.get();
        offset += block_size;
    }

    if (matched_tokens != nullptr) {
        *matched_tokens = best_tokens;
    }
    return best_id;
}

size_t server_prefix_cache_index::size() const {
    return records.size();
}
