#include "llama-paged-kv-pool.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>

static constexpr uint32_t LLAMA_PAGED_KV_STATE_MAGIC = 0x51504b56;
static constexpr uint32_t LLAMA_PAGED_KV_SEQUENCE_STATE_VERSION = 1;
static constexpr uint32_t LLAMA_PAGED_KV_FULL_STATE_VERSION = 2;

llama_paged_kv_pool::llama_paged_kv_pool(
        const llama_model & model,
                  ggml_type type,
                   uint32_t page_size_value,
                   uint32_t n_pages_value,
                       bool offload,
        const llama_memory_i::layer_filter_cb & filter) :
    block_size(page_size_value),
    block_count(n_pages_value) {
    if (type != GGML_TYPE_Q8_0 || block_size == 0 || block_count == 0) {
        throw std::runtime_error("invalid Qwen3.8 paged KV configuration");
    }

    struct buft_less {
        bool operator()(ggml_backend_buffer_type_t lhs, ggml_backend_buffer_type_t rhs) const {
            return std::strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, buft_less> context_map;

    auto context_for = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = context_map.find(buft);
        if (it != context_map.end()) {
            return it->second.get();
        }
        ggml_init_params params = {
            /*.mem_size   =*/ static_cast<size_t>(2 * model.hparams.n_layer_all) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        if (ctx == nullptr) {
            return nullptr;
        }
        context_map.emplace(buft, ctx);
        return ctx;
    };

    for (uint32_t il = 0; il < model.hparams.n_layer_all; ++il) {
        if (!model.hparams.has_kv(il) || (filter && !filter(il))) {
            continue;
        }

        ggml_backend_dev_t dev = offload ? model.dev_layer(il) : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (dev == nullptr) {
            throw std::runtime_error("paged KV layer has no backend device");
        }
        auto device = std::find(devices.begin(), devices.end(), dev);
        if (device == devices.end()) {
            devices.push_back(dev);
            device = std::prev(devices.end());
        }
        const uint32_t device_id = static_cast<uint32_t>(std::distance(devices.begin(), device));
        if (device_id >= 2) {
            throw std::runtime_error("Qwen3.8 paged KV requires at most two devices");
        }

        ggml_backend_buffer_type_t buft = offload ? ggml_backend_dev_buffer_type(dev) : ggml_backend_cpu_buffer_type();
        ggml_context * ctx = context_for(buft);
        if (ctx == nullptr) {
            throw std::runtime_error("failed to create paged KV tensor context");
        }

        const uint32_t head_dim_k = model.hparams.n_embd_head_k(il);
        const uint32_t head_dim_v = model.hparams.n_embd_head_v(il);
        const uint32_t n_head_kv = model.hparams.n_head_kv(il);
        if (head_dim_k != head_dim_v || head_dim_k % ggml_blck_size(type) != 0) {
            throw std::runtime_error("unsupported paged KV head layout");
        }

        ggml_tensor * kv = ggml_new_tensor_4d(ctx, type, head_dim_k, block_size, 2 * n_head_kv, block_count);
        ggml_format_name(kv, "paged_kv_l%d", il);
        layer_map[il] = layers.size();
        layers.push_back({ static_cast<int32_t>(il), device_id, kv });
    }

    for (auto & entry : context_map) {
        ggml_backend_buffer_t buffer;
        if (model.hparams.no_alloc) {
            buffer = ggml_backend_buft_alloc_buffer(entry.first, 0);
            for (ggml_tensor * tensor = ggml_get_first_tensor(entry.second.get()); tensor != nullptr; tensor = ggml_get_next_tensor(entry.second.get(), tensor)) {
                tensor->buffer = buffer;
            }
        } else {
            buffer = ggml_backend_alloc_ctx_tensors_from_buft(entry.second.get(), entry.first);
        }
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate paged KV buffer");
        }
        ggml_backend_buffer_clear(buffer, 0);
        LLAMA_LOG_INFO("%s: %10s paged KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buffer), ggml_backend_buffer_get_size(buffer) / 1024.0 / 1024.0);
        contexts.emplace_back(std::move(entry.second), buffer);
    }
}

ggml_tensor * llama_paged_kv_pool::get_kv(int32_t il) const {
    const auto it = layer_map.find(il);
    return it == layer_map.end() ? nullptr : layers[it->second].kv;
}

uint32_t llama_paged_kv_pool::device_index(int32_t il) const {
    const auto it = layer_map.find(il);
    if (it == layer_map.end()) {
        throw std::out_of_range("paged KV layer is not present");
    }
    return layers[it->second].device;
}

uint32_t llama_paged_kv_pool::n_devices() const {
    return devices.size();
}

uint32_t llama_paged_kv_pool::page_size() const {
    return block_size;
}

uint32_t llama_paged_kv_pool::n_pages() const {
    return block_count;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_paged_kv_pool::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> result;
    for (const auto & value : contexts) {
        ggml_backend_buffer_t buffer = value.second.get();
        result[ggml_backend_buffer_get_type(buffer)] += ggml_backend_buffer_get_size(buffer);
    }
    return result;
}

void llama_paged_kv_pool::clear() {
    for (const auto & value : contexts) {
        ggml_backend_buffer_clear(value.second.get(), 0);
    }
}

void llama_paged_kv_pool::copy_pages(const std::vector<llama_paged_block_copy> & copies) const {
    for (const auto & copy : copies) {
        for (const auto & value : layers) {
            const size_t page_bytes = value.kv->nb[3];
            std::vector<uint8_t> staging(page_bytes);
            const size_t source_offset = static_cast<size_t>(copy.source.physical[value.device]) * page_bytes;
            const size_t destination_offset = static_cast<size_t>(copy.destination.physical[value.device]) * page_bytes;
            ggml_backend_tensor_get(value.kv, staging.data(), source_offset, page_bytes);
            ggml_backend_tensor_set(value.kv, staging.data(), destination_offset, page_bytes);
        }
    }
}

void llama_paged_kv_pool::state_write(
        llama_io_write_i & io,
        const llama_paged_block_manager & manager,
        int sequence_id) const {
    if (sequence_id < 0) {
        auto sequence_ids = manager.sequence_ids();
        std::sort(sequence_ids.begin(), sequence_ids.end());
        std::vector<std::vector<llama_paged_block_handle>> tables;
        std::vector<uint64_t> token_counts;
        std::vector<llama_paged_block_handle> unique_pages;
        std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, uint32_t> unique_indices;
        tables.reserve(sequence_ids.size());
        token_counts.reserve(sequence_ids.size());
        for (int id : sequence_ids) {
            tables.push_back(manager.page_table(id));
            token_counts.push_back(manager.token_count(id));
            for (const auto & page : tables.back()) {
                const auto key = std::make_tuple(page.id, page.generation, page.physical[0], page.physical[1]);
                if (unique_indices.find(key) == unique_indices.end()) {
                    const uint32_t index = unique_pages.size();
                    unique_indices.emplace(key, index);
                    unique_pages.push_back(page);
                }
            }
        }
        const uint32_t sequence_count = sequence_ids.size();
        const uint32_t unique_page_count = unique_pages.size();
        const uint32_t layer_count = layers.size();
        io.write(&LLAMA_PAGED_KV_STATE_MAGIC, sizeof(LLAMA_PAGED_KV_STATE_MAGIC));
        io.write(&LLAMA_PAGED_KV_FULL_STATE_VERSION, sizeof(LLAMA_PAGED_KV_FULL_STATE_VERSION));
        io.write(&block_size, sizeof(block_size));
        io.write(&sequence_count, sizeof(sequence_count));
        io.write(&unique_page_count, sizeof(unique_page_count));
        io.write(&layer_count, sizeof(layer_count));
        for (const auto & value : layers) {
            const uint64_t page_bytes = value.kv->nb[3];
            io.write(&value.il, sizeof(value.il));
            io.write(&page_bytes, sizeof(page_bytes));
        }
        for (size_t sequence = 0; sequence < sequence_ids.size(); ++sequence) {
            const int32_t id = sequence_ids[sequence];
            const uint64_t tokens = token_counts[sequence];
            const uint32_t page_count = tables[sequence].size();
            io.write(&id, sizeof(id));
            io.write(&tokens, sizeof(tokens));
            io.write(&page_count, sizeof(page_count));
            for (const auto & page : tables[sequence]) {
                const auto key = std::make_tuple(page.id, page.generation, page.physical[0], page.physical[1]);
                const uint32_t index = unique_indices.at(key);
                io.write(&index, sizeof(index));
            }
        }
        for (const auto & value : layers) {
            const uint64_t page_bytes = value.kv->nb[3];
            for (const auto & page : unique_pages) {
                io.write_tensor(value.kv, static_cast<size_t>(page.physical[value.device]) * page_bytes, page_bytes);
            }
        }
        return;
    }
    const auto table = manager.page_table(sequence_id);
    const uint64_t tokens = manager.token_count(sequence_id);
    const uint32_t page_count = table.size();
    const uint32_t layer_count = layers.size();
    io.write(&LLAMA_PAGED_KV_STATE_MAGIC, sizeof(LLAMA_PAGED_KV_STATE_MAGIC));
    io.write(&LLAMA_PAGED_KV_SEQUENCE_STATE_VERSION, sizeof(LLAMA_PAGED_KV_SEQUENCE_STATE_VERSION));
    io.write(&block_size, sizeof(block_size));
    io.write(&tokens, sizeof(tokens));
    io.write(&page_count, sizeof(page_count));
    io.write(&layer_count, sizeof(layer_count));
    for (const auto & value : layers) {
        const uint64_t page_bytes = value.kv->nb[3];
        io.write(&value.il, sizeof(value.il));
        io.write(&page_bytes, sizeof(page_bytes));
        for (const auto & page : table) {
            io.write_tensor(value.kv, static_cast<size_t>(page.physical[value.device]) * page_bytes, page_bytes);
        }
    }
}

void llama_paged_kv_pool::state_read(
        llama_io_read_i & io,
        llama_paged_block_manager & manager,
        int sequence_id) const {
    uint32_t magic;
    uint32_t version;
    io.read(&magic, sizeof(magic));
    io.read(&version, sizeof(version));
    if (magic != LLAMA_PAGED_KV_STATE_MAGIC) {
        throw std::runtime_error("paged KV state magic mismatch");
    }

    if (sequence_id < 0) {
        uint32_t saved_block_size;
        uint32_t sequence_count;
        uint32_t unique_page_count;
        uint32_t layer_count;
        io.read(&saved_block_size, sizeof(saved_block_size));
        io.read(&sequence_count, sizeof(sequence_count));
        io.read(&unique_page_count, sizeof(unique_page_count));
        io.read(&layer_count, sizeof(layer_count));
        if (version != LLAMA_PAGED_KV_FULL_STATE_VERSION || saved_block_size != block_size ||
                sequence_count > manager.max_sequences() || unique_page_count > block_count || layer_count != layers.size()) {
            throw std::runtime_error("paged KV full-state metadata mismatch");
        }
        for (const auto & value : layers) {
            int32_t saved_layer;
            uint64_t page_bytes;
            io.read(&saved_layer, sizeof(saved_layer));
            io.read(&page_bytes, sizeof(page_bytes));
            if (saved_layer != value.il || page_bytes != value.kv->nb[3]) {
                throw std::runtime_error("paged KV full-state layer metadata mismatch");
            }
        }

        std::vector<llama_paged_sequence_layout> layouts;
        layouts.reserve(sequence_count);
        for (uint32_t sequence = 0; sequence < sequence_count; ++sequence) {
            int32_t id;
            uint64_t tokens;
            uint32_t page_count;
            io.read(&id, sizeof(id));
            io.read(&tokens, sizeof(tokens));
            io.read(&page_count, sizeof(page_count));
            const uint64_t max_tokens = static_cast<uint64_t>(block_size) * block_count;
            if (id < 0 || tokens == 0 || tokens > max_tokens || tokens > std::numeric_limits<size_t>::max() ||
                    page_count != (tokens + block_size - 1) / block_size || page_count > unique_page_count) {
                throw std::runtime_error("paged KV full-state sequence metadata mismatch");
            }
            llama_paged_sequence_layout layout;
            layout.sequence_id = id;
            layout.tokens = tokens;
            layout.page_indices.resize(page_count);
            for (uint32_t & index : layout.page_indices) {
                io.read(&index, sizeof(index));
            }
            layouts.push_back(std::move(layout));
        }

        manager.clear();
        std::vector<llama_paged_block_handle> restored_pages;
        if (!manager.restore_layout(layouts, unique_page_count, restored_pages)) {
            throw std::runtime_error("failed to restore paged KV shared layout");
        }
        try {
            for (const auto & value : layers) {
                const uint64_t page_bytes = value.kv->nb[3];
                for (const auto & page : restored_pages) {
                    io.read_tensor(value.kv, static_cast<size_t>(page.physical[value.device]) * page_bytes, page_bytes);
                }
            }
        } catch (...) {
            manager.clear();
            throw;
        }
        return;
    }

    uint32_t saved_block_size;
    uint64_t tokens;
    uint32_t page_count;
    uint32_t layer_count;
    io.read(&saved_block_size, sizeof(saved_block_size));
    io.read(&tokens, sizeof(tokens));
    io.read(&page_count, sizeof(page_count));
    io.read(&layer_count, sizeof(layer_count));
    if (version != LLAMA_PAGED_KV_SEQUENCE_STATE_VERSION || saved_block_size != block_size || layer_count != layers.size()) {
        throw std::runtime_error("paged KV state metadata mismatch");
    }
    const uint64_t max_tokens = static_cast<uint64_t>(block_size) * block_count;
    if (tokens > max_tokens) {
        throw std::runtime_error("paged KV state size mismatch");
    }
    const uint32_t expected_pages = static_cast<uint32_t>((tokens + block_size - 1) / block_size);
    if (page_count != expected_pages) {
        throw std::runtime_error("paged KV state size mismatch");
    }

    if (!manager.reserve(sequence_id, 0) || !manager.reserve(sequence_id, tokens)) {
        throw std::runtime_error("failed to allocate paged KV restore pages");
    }
    const auto table = manager.page_table(sequence_id);
    try {
        for (const auto & value : layers) {
            int32_t saved_layer;
            uint64_t page_bytes;
            io.read(&saved_layer, sizeof(saved_layer));
            io.read(&page_bytes, sizeof(page_bytes));
            if (saved_layer != value.il || page_bytes != value.kv->nb[3]) {
                throw std::runtime_error("paged KV layer metadata mismatch");
            }
            for (const auto & page : table) {
                io.read_tensor(value.kv, static_cast<size_t>(page.physical[value.device]) * page_bytes, page_bytes);
            }
        }
    } catch (...) {
        manager.reserve(sequence_id, 0);
        throw;
    }
}
