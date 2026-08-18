#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <utility>
#include <unordered_map>
#include <vector>

struct llama_paged_block_handle {
    uint32_t id = 0;
    uint32_t generation = 0;
    std::array<uint32_t, 2> physical = { 0, 0 };

    bool operator==(const llama_paged_block_handle & other) const {
        return id == other.id && generation == other.generation && physical == other.physical;
    }
};

struct llama_paged_block_manager_stats {
    size_t total_logical_pages = 0;
    size_t free_logical_pages = 0;
    std::array<size_t, 2> total_physical_pages = { 0, 0 };
    std::array<size_t, 2> free_physical_pages = { 0, 0 };
    size_t resident_sequences = 0;
    size_t admitted_pages = 0;
    size_t committed_pages = 0;
    size_t cow_pages = 0;
    size_t allocation_failures = 0;
    size_t cached_prefixes = 0;
    size_t cached_prefix_pages = 0;
};

struct llama_paged_block_copy {
    llama_paged_block_handle source;
    llama_paged_block_handle destination;
};

struct llama_paged_sequence_layout {
    int sequence_id = -1;
    size_t tokens = 0;
    std::vector<uint32_t> page_indices;
};

struct llama_paged_prefix_handle {
    uint64_t id = 0;

    bool operator==(const llama_paged_prefix_handle & other) const {
        return id == other.id;
    }
};

class llama_paged_block_manager {
public:
    void configure(size_t total_tokens, size_t page_size, size_t max_sequences);
    void configure_physical(const std::array<size_t, 2> & physical_pages, size_t page_size, size_t max_sequences);

    bool reserve(int sequence_id, size_t tokens);
    bool can_reserve_batch(const std::vector<std::pair<int, size_t>> & requests) const;
    bool reserve_batch(const std::vector<std::pair<int, size_t>> & requests);
    bool fork_sequence(int source_id, int destination_id);
    bool pin_prefix(int source_id, size_t tokens, llama_paged_prefix_handle & prefix);
    bool attach_prefix(int destination_id, const llama_paged_prefix_handle & prefix);
    bool release_prefix(const llama_paged_prefix_handle & prefix);
    bool ensure_writable_tail(int sequence_id);
    bool admit(int sequence_id, size_t max_tokens);
    void release_admission(int sequence_id);
    void release(int sequence_id);
    void keep(int sequence_id);
    void clear();

    size_t page_size() const;
    size_t max_sequences() const;
    size_t reserved_tokens(int sequence_id) const;
    size_t token_count(int sequence_id) const;
    size_t prefix_token_count(const llama_paged_prefix_handle & prefix) const;
    std::vector<llama_paged_block_handle> page_table(int sequence_id) const;
    std::vector<int> sequence_ids() const;
    bool restore_layout(
        const std::vector<llama_paged_sequence_layout> & layouts,
        size_t unique_page_count,
        std::vector<llama_paged_block_handle> & restored_pages);
    std::vector<llama_paged_block_copy> take_pending_copies();
    llama_paged_block_manager_stats stats() const;

private:
    struct logical_page {
        uint32_t generation = 1;
        uint32_t refs = 0;
        std::array<uint32_t, 2> physical = { 0, 0 };
    };

    struct sequence_state {
        std::vector<llama_paged_block_handle> pages;
        size_t tokens = 0;
        size_t quota_pages = 0;
    };

    struct prefix_state {
        std::vector<llama_paged_block_handle> pages;
        size_t tokens = 0;
    };

    bool allocate_page(llama_paged_block_handle & handle);
    void reserve_locked(int sequence_id, size_t tokens);
    bool can_reserve_batch_locked(const std::vector<std::pair<int, size_t>> & requests) const;
    void release_page(const llama_paged_block_handle & handle);
    bool valid(const llama_paged_block_handle & handle) const;
    size_t available_pages() const;
    size_t committed_pages_locked() const;

    mutable std::mutex mutex;
    size_t block_size = 0;
    size_t sequence_limit = 0;
    size_t cow_count = 0;
    size_t failure_count = 0;
    std::vector<logical_page> logical_pages;
    std::vector<uint32_t> free_logical_ids;
    std::array<std::vector<uint32_t>, 2> free_physical_ids;
    std::array<size_t, 2> physical_totals = { 0, 0 };
    std::unordered_map<int, sequence_state> sequences;
    std::unordered_map<uint64_t, prefix_state> prefixes;
    uint64_t next_prefix_id = 1;
    std::vector<llama_paged_block_copy> pending_copies;
};
