#pragma once

#include "llama-memory.h"
#include "llama-paged-block-manager.h"

#include <array>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

class llama_model;
class llama_io_write_i;
class llama_io_read_i;

struct llama_paged_kv_metadata {
    std::array<std::vector<int32_t>, 2> write_slots;
    std::array<std::vector<int32_t>, 2> block_tables;
    std::vector<int32_t> context_lens;
    std::vector<int32_t> batch_offsets;
    std::vector<int32_t> batch_lens;
    int32_t n_sequences = 0;
    int32_t max_blocks = 0;
};

class llama_paged_kv_pool {
public:
    llama_paged_kv_pool(
        const llama_model & model,
                  ggml_type type,
                   uint32_t page_size,
                   uint32_t n_pages,
                       bool offload,
        const llama_memory_i::layer_filter_cb & filter);

    ggml_tensor * get_kv(int32_t il) const;
    uint32_t device_index(int32_t il) const;
    uint32_t n_devices() const;
    uint32_t page_size() const;
    uint32_t n_pages() const;
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;
    void clear();
    void copy_pages(const std::vector<llama_paged_block_copy> & copies) const;
    void state_write(llama_io_write_i & io, const llama_paged_block_manager & manager, int sequence_id) const;
    void state_read(llama_io_read_i & io, llama_paged_block_manager & manager, int sequence_id) const;

private:
    struct layer {
        int32_t il = -1;
        uint32_t device = 0;
        ggml_tensor * kv = nullptr;
    };

    uint32_t block_size = 0;
    uint32_t block_count = 0;
    std::vector<ggml_backend_dev_t> devices;
    std::vector<layer> layers;
    std::map<int32_t, size_t> layer_map;
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> contexts;
};
