#include "llama-paged-block-manager.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <unordered_set>

void llama_paged_block_manager::configure(size_t total_tokens, size_t page_size_value, size_t max_sequences) {
    const size_t n_pages = page_size_value == 0 ? 0 : total_tokens / page_size_value;
    configure_physical({ n_pages, n_pages }, page_size_value, max_sequences);
}

void llama_paged_block_manager::configure_physical(
        const std::array<size_t, 2> & physical_pages,
        size_t page_size_value,
        size_t max_sequences) {
    std::lock_guard<std::mutex> lock(mutex);
    block_size = page_size_value;
    sequence_limit = max_sequences;
    cow_count = 0;
    failure_count = 0;
    sequences.clear();
    prefixes.clear();
    pending_copies.clear();
    const size_t id_limit = std::numeric_limits<uint32_t>::max();
    physical_totals = {
        std::min(physical_pages[0], id_limit),
        std::min(physical_pages[1], id_limit),
    };

    const size_t logical_count = std::min(physical_totals[0], physical_totals[1]);
    logical_pages.assign(logical_count, {});
    free_logical_ids.clear();
    free_logical_ids.reserve(logical_count);
    for (size_t i = logical_count; i > 0; --i) {
        free_logical_ids.push_back(static_cast<uint32_t>(i - 1));
    }

    for (size_t device = 0; device < free_physical_ids.size(); ++device) {
        auto & ids = free_physical_ids[device];
        ids.clear();
        ids.reserve(physical_totals[device]);
        for (size_t i = physical_totals[device]; i > 0; --i) {
            ids.push_back(static_cast<uint32_t>(i - 1));
        }
    }
}

size_t llama_paged_block_manager::available_pages() const {
    return std::min({ free_logical_ids.size(), free_physical_ids[0].size(), free_physical_ids[1].size() });
}

size_t llama_paged_block_manager::committed_pages_locked() const {
    size_t result = 0;
    for (const auto & sequence : sequences) {
        result += sequence.second.pages.size();
    }
    return result;
}

bool llama_paged_block_manager::allocate_page(llama_paged_block_handle & handle) {
    if (available_pages() == 0) {
        return false;
    }

    const uint32_t id = free_logical_ids.back();
    free_logical_ids.pop_back();

    logical_page & page = logical_pages[id];
    page.refs = 1;
    for (size_t device = 0; device < free_physical_ids.size(); ++device) {
        page.physical[device] = free_physical_ids[device].back();
        free_physical_ids[device].pop_back();
    }
    handle = { id, page.generation, page.physical };
    return true;
}

bool llama_paged_block_manager::valid(const llama_paged_block_handle & handle) const {
    if (handle.id >= logical_pages.size()) {
        return false;
    }
    const logical_page & page = logical_pages[handle.id];
    return page.generation == handle.generation && page.refs > 0 && page.physical == handle.physical;
}

void llama_paged_block_manager::release_page(const llama_paged_block_handle & handle) {
    if (!valid(handle)) {
        return;
    }

    logical_page & page = logical_pages[handle.id];
    --page.refs;
    if (page.refs != 0) {
        return;
    }

    for (size_t device = 0; device < free_physical_ids.size(); ++device) {
        free_physical_ids[device].push_back(page.physical[device]);
    }
    ++page.generation;
    if (page.generation == 0) {
        page.generation = 1;
    }
    free_logical_ids.push_back(handle.id);
}

bool llama_paged_block_manager::reserve(int sequence_id, size_t tokens) {
    return reserve_batch({ { sequence_id, tokens } });
}

bool llama_paged_block_manager::can_reserve_batch_locked(const std::vector<std::pair<int, size_t>> & requests) const {
    if (block_size == 0) {
        return false;
    }

    size_t required_pages = 0;
    size_t new_sequences = 0;
    std::unordered_set<int> seen;
    for (const auto & request : requests) {
        const int sequence_id = request.first;
        const size_t tokens = request.second;
        if (sequence_id < 0 || !seen.insert(sequence_id).second || tokens > std::numeric_limits<size_t>::max() - (block_size - 1)) {
            return false;
        }

        const auto it = sequences.find(sequence_id);
        if (it == sequences.end()) {
            ++new_sequences;
        }

        const size_t current_pages = it == sequences.end() ? 0 : it->second.pages.size();
        const size_t current_tokens = it == sequences.end() ? 0 : it->second.tokens;
        const size_t requested_pages = (tokens + block_size - 1) / block_size;
        const size_t additional = requested_pages > current_pages ? requested_pages - current_pages : 0;

        bool needs_cow = false;
        if (it != sequences.end() && tokens > current_tokens && current_tokens % block_size != 0 && !it->second.pages.empty()) {
            const auto & tail = it->second.pages.back();
            if (!valid(tail)) {
                return false;
            }
            needs_cow = logical_pages[tail.id].refs > 1;
        }
        const size_t cow_pages = needs_cow ? 1 : 0;
        if (additional > std::numeric_limits<size_t>::max() - cow_pages) {
            return false;
        }
        const size_t needed = additional + cow_pages;
        if (required_pages > std::numeric_limits<size_t>::max() - needed) {
            return false;
        }
        required_pages += needed;

    }

    return sequences.size() + new_sequences <= sequence_limit && required_pages <= available_pages();
}

bool llama_paged_block_manager::can_reserve_batch(const std::vector<std::pair<int, size_t>> & requests) const {
    std::lock_guard<std::mutex> lock(mutex);
    return can_reserve_batch_locked(requests);
}

void llama_paged_block_manager::reserve_locked(int sequence_id, size_t tokens) {
    auto it = sequences.find(sequence_id);
    if (it == sequences.end()) {
        it = sequences.emplace(sequence_id, sequence_state{}).first;
    }

    auto & state = it->second;
    auto & handles = state.pages;
    const size_t required = (tokens + block_size - 1) / block_size;
    const bool needs_cow = tokens > state.tokens && state.tokens % block_size != 0 && !handles.empty() &&
        logical_pages[handles.back().id].refs > 1;

    if (needs_cow) {
        const llama_paged_block_handle source = handles.back();
        llama_paged_block_handle replacement;
        if (!allocate_page(replacement)) {
            std::abort();
        }
        release_page(handles.back());
        handles.back() = replacement;
        pending_copies.push_back({ source, replacement });
        ++cow_count;
    }

    while (handles.size() < required) {
        llama_paged_block_handle handle;
        if (!allocate_page(handle)) {
            std::abort();
        }
        handles.push_back(handle);
    }
    while (handles.size() > required) {
        release_page(handles.back());
        handles.pop_back();
    }
    state.tokens = tokens;
    if (handles.empty() && state.quota_pages == 0) {
        sequences.erase(sequence_id);
    }
}

bool llama_paged_block_manager::reserve_batch(const std::vector<std::pair<int, size_t>> & requests) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!can_reserve_batch_locked(requests)) {
        ++failure_count;
        return false;
    }
    for (const auto & request : requests) {
        reserve_locked(request.first, request.second);
    }
    return true;
}

bool llama_paged_block_manager::fork_sequence(int source_id, int destination_id) {
    std::lock_guard<std::mutex> lock(mutex);
    const auto source = sequences.find(source_id);
    const auto destination = sequences.find(destination_id);
    if (source == sequences.end() || (destination != sequences.end() && !destination->second.pages.empty()) ||
            (destination == sequences.end() && sequences.size() >= sequence_limit)) {
        ++failure_count;
        return false;
    }

    auto state = source->second;
    state.quota_pages = destination == sequences.end() ? 0 : destination->second.quota_pages;
    for (const auto & handle : state.pages) {
        if (!valid(handle)) {
            ++failure_count;
            return false;
        }
    }
    for (const auto & handle : state.pages) {
        ++logical_pages[handle.id].refs;
    }
    if (destination == sequences.end()) {
        sequences.emplace(destination_id, std::move(state));
    } else {
        destination->second = std::move(state);
    }
    return true;
}

bool llama_paged_block_manager::pin_prefix(int source_id, size_t tokens, llama_paged_prefix_handle & prefix) {
    std::lock_guard<std::mutex> lock(mutex);
    const auto source = sequences.find(source_id);
    if (block_size == 0 || prefix.id != 0 || tokens == 0 || source == sequences.end() || tokens > source->second.tokens ||
            tokens > std::numeric_limits<size_t>::max() - (block_size - 1)) {
        ++failure_count;
        return false;
    }

    const size_t page_count = (tokens + block_size - 1) / block_size;
    if (page_count > source->second.pages.size()) {
        ++failure_count;
        return false;
    }
    for (size_t i = 0; i < page_count; ++i) {
        if (!valid(source->second.pages[i])) {
            ++failure_count;
            return false;
        }
    }

    uint64_t id = next_prefix_id++;
    if (id == 0) {
        id = next_prefix_id++;
    }
    prefix_state state;
    state.pages.assign(source->second.pages.begin(), source->second.pages.begin() + page_count);
    state.tokens = tokens;
    for (const auto & handle : state.pages) {
        ++logical_pages[handle.id].refs;
    }
    prefixes.emplace(id, std::move(state));
    prefix.id = id;
    return true;
}

bool llama_paged_block_manager::attach_prefix(int destination_id, const llama_paged_prefix_handle & prefix) {
    std::lock_guard<std::mutex> lock(mutex);
    const auto cached = prefixes.find(prefix.id);
    auto destination = sequences.find(destination_id);
    if (destination_id < 0 || prefix.id == 0 || cached == prefixes.end() ||
            (destination != sequences.end() && (!destination->second.pages.empty() || destination->second.tokens != 0)) ||
            (destination == sequences.end() && sequences.size() >= sequence_limit)) {
        ++failure_count;
        return false;
    }
    for (const auto & handle : cached->second.pages) {
        if (!valid(handle)) {
            ++failure_count;
            return false;
        }
    }

    sequence_state state;
    state.pages = cached->second.pages;
    state.tokens = cached->second.tokens;
    state.quota_pages = destination == sequences.end() ? 0 : destination->second.quota_pages;
    for (const auto & handle : state.pages) {
        ++logical_pages[handle.id].refs;
    }
    if (destination == sequences.end()) {
        sequences.emplace(destination_id, std::move(state));
    } else {
        destination->second = std::move(state);
    }
    return true;
}

bool llama_paged_block_manager::release_prefix(const llama_paged_prefix_handle & prefix) {
    std::lock_guard<std::mutex> lock(mutex);
    const auto it = prefixes.find(prefix.id);
    if (prefix.id == 0 || it == prefixes.end()) {
        return false;
    }
    for (const auto & handle : it->second.pages) {
        release_page(handle);
    }
    prefixes.erase(it);
    return true;
}

bool llama_paged_block_manager::ensure_writable_tail(int sequence_id) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = sequences.find(sequence_id);
    if (it == sequences.end() || it->second.pages.empty()) {
        return true;
    }

    llama_paged_block_handle & tail = it->second.pages.back();
    if (!valid(tail)) {
        ++failure_count;
        return false;
    }
    if (logical_pages[tail.id].refs == 1) {
        return true;
    }

    llama_paged_block_handle replacement;
    if (!allocate_page(replacement)) {
        ++failure_count;
        return false;
    }
    const llama_paged_block_handle source = tail;
    release_page(tail);
    tail = replacement;
    pending_copies.push_back({ source, replacement });
    ++cow_count;
    return true;
}

bool llama_paged_block_manager::admit(int sequence_id, size_t max_tokens) {
    std::lock_guard<std::mutex> lock(mutex);
    if (block_size == 0 || sequence_id < 0 || max_tokens > std::numeric_limits<size_t>::max() - (block_size - 1)) {
        ++failure_count;
        return false;
    }

    auto it = sequences.find(sequence_id);
    if (it == sequences.end()) {
        if (sequences.size() >= sequence_limit) {
            ++failure_count;
            return false;
        }
        it = sequences.emplace(sequence_id, sequence_state{}).first;
    }

    const size_t requested_quota = (max_tokens + block_size - 1) / block_size;
    if (requested_quota > logical_pages.size()) {
        ++failure_count;
        if (it->second.pages.empty() && it->second.quota_pages == 0) {
            sequences.erase(it);
        }
        return false;
    }

    it->second.quota_pages = requested_quota;
    if (it->second.pages.empty() && requested_quota == 0) {
        sequences.erase(it);
    }
    return true;
}

void llama_paged_block_manager::release_admission(int sequence_id) {
    std::lock_guard<std::mutex> lock(mutex);
    const auto it = sequences.find(sequence_id);
    if (it == sequences.end()) {
        return;
    }
    it->second.quota_pages = 0;
    if (it->second.pages.empty()) {
        sequences.erase(it);
    }
}

void llama_paged_block_manager::release(int sequence_id) {
    std::lock_guard<std::mutex> lock(mutex);
    const auto it = sequences.find(sequence_id);
    if (it == sequences.end()) {
        return;
    }
    for (const auto & handle : it->second.pages) {
        release_page(handle);
    }
    sequences.erase(it);
}

void llama_paged_block_manager::keep(int sequence_id) {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto it = sequences.begin(); it != sequences.end();) {
        if (it->first == sequence_id) {
            ++it;
            continue;
        }
        for (const auto & handle : it->second.pages) {
            release_page(handle);
        }
        it = sequences.erase(it);
    }
}

void llama_paged_block_manager::clear() {
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto & sequence : sequences) {
        for (const auto & handle : sequence.second.pages) {
            release_page(handle);
        }
    }
    sequences.clear();
    for (const auto & prefix : prefixes) {
        for (const auto & handle : prefix.second.pages) {
            release_page(handle);
        }
    }
    prefixes.clear();
    pending_copies.clear();
}

size_t llama_paged_block_manager::page_size() const {
    std::lock_guard<std::mutex> lock(mutex);
    return block_size;
}

size_t llama_paged_block_manager::max_sequences() const {
    std::lock_guard<std::mutex> lock(mutex);
    return sequence_limit;
}

size_t llama_paged_block_manager::reserved_tokens(int sequence_id) const {
    std::lock_guard<std::mutex> lock(mutex);
    const auto it = sequences.find(sequence_id);
    return it == sequences.end() ? 0 : it->second.pages.size() * block_size;
}

size_t llama_paged_block_manager::token_count(int sequence_id) const {
    std::lock_guard<std::mutex> lock(mutex);
    const auto it = sequences.find(sequence_id);
    return it == sequences.end() ? 0 : it->second.tokens;
}

size_t llama_paged_block_manager::prefix_token_count(const llama_paged_prefix_handle & prefix) const {
    std::lock_guard<std::mutex> lock(mutex);
    const auto it = prefixes.find(prefix.id);
    return it == prefixes.end() ? 0 : it->second.tokens;
}

std::vector<llama_paged_block_handle> llama_paged_block_manager::page_table(int sequence_id) const {
    std::lock_guard<std::mutex> lock(mutex);
    const auto it = sequences.find(sequence_id);
    return it == sequences.end() ? std::vector<llama_paged_block_handle>{} : it->second.pages;
}

std::vector<int> llama_paged_block_manager::sequence_ids() const {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<int> result;
    result.reserve(sequences.size());
    for (const auto & sequence : sequences) {
        if (!sequence.second.pages.empty()) {
            result.push_back(sequence.first);
        }
    }
    return result;
}

bool llama_paged_block_manager::restore_layout(
        const std::vector<llama_paged_sequence_layout> & layouts,
        size_t unique_page_count,
        std::vector<llama_paged_block_handle> & restored_pages) {
    std::lock_guard<std::mutex> lock(mutex);
    restored_pages.clear();
    if (block_size == 0 || !sequences.empty() || !prefixes.empty() || layouts.size() > sequence_limit ||
            unique_page_count > available_pages()) {
        ++failure_count;
        return false;
    }

    std::unordered_set<int> sequence_ids_seen;
    std::vector<bool> referenced(unique_page_count, false);
    for (const auto & layout : layouts) {
        if (layout.sequence_id < 0 || layout.tokens == 0 || !sequence_ids_seen.insert(layout.sequence_id).second ||
                layout.tokens > std::numeric_limits<size_t>::max() - (block_size - 1) ||
                layout.page_indices.size() != (layout.tokens + block_size - 1) / block_size) {
            ++failure_count;
            return false;
        }
        std::unordered_set<uint32_t> page_indices_seen;
        for (uint32_t index : layout.page_indices) {
            if (index >= unique_page_count || !page_indices_seen.insert(index).second) {
                ++failure_count;
                return false;
            }
            referenced[index] = true;
        }
    }
    if (std::find(referenced.begin(), referenced.end(), false) != referenced.end()) {
        ++failure_count;
        return false;
    }

    restored_pages.resize(unique_page_count);
    for (auto & handle : restored_pages) {
        if (!allocate_page(handle)) {
            std::abort();
        }
        logical_pages[handle.id].refs = 0;
    }
    try {
        for (const auto & layout : layouts) {
            sequence_state state;
            state.tokens = layout.tokens;
            state.pages.reserve(layout.page_indices.size());
            for (uint32_t index : layout.page_indices) {
                const auto & handle = restored_pages[index];
                state.pages.push_back(handle);
                ++logical_pages[handle.id].refs;
            }
            sequences.emplace(layout.sequence_id, std::move(state));
        }
    } catch (...) {
        for (const auto & sequence : sequences) {
            for (const auto & handle : sequence.second.pages) {
                release_page(handle);
            }
        }
        sequences.clear();
        for (const auto & handle : restored_pages) {
            auto & page = logical_pages[handle.id];
            if (page.generation == handle.generation && page.refs == 0 && page.physical == handle.physical) {
                page.refs = 1;
                release_page(handle);
            }
        }
        restored_pages.clear();
        throw;
    }
    return true;
}

std::vector<llama_paged_block_copy> llama_paged_block_manager::take_pending_copies() {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<llama_paged_block_copy> result = std::move(pending_copies);
    pending_copies.clear();
    return result;
}

llama_paged_block_manager_stats llama_paged_block_manager::stats() const {
    std::lock_guard<std::mutex> lock(mutex);
    size_t admitted_pages = 0;
    size_t cached_prefix_pages = 0;
    for (const auto & sequence : sequences) {
        admitted_pages += sequence.second.quota_pages;
    }
    for (const auto & prefix : prefixes) {
        cached_prefix_pages += prefix.second.pages.size();
    }
    return {
        logical_pages.size(),
        free_logical_ids.size(),
        physical_totals,
        { free_physical_ids[0].size(), free_physical_ids[1].size() },
        sequences.size(),
        admitted_pages,
        committed_pages_locked(),
        cow_count,
        failure_count,
        prefixes.size(),
        cached_prefix_pages,
    };
}
