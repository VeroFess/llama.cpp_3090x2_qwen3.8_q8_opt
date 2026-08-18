#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (device == nullptr) {
        fprintf(stderr, "%s : skipping because no GPU backend is available\n", __func__);
        return 0;
    }
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (backend == nullptr) {
        fprintf(stderr, "%s : failed to initialize GPU backend\n", __func__);
        return 1;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);

    constexpr int head_dim = 256;
    constexpr int n_heads = 24;
    constexpr int n_heads_kv = 4;
    constexpr int block_size = 16;
    constexpr int n_blocks = 6;
    constexpr int max_blocks = 5;

    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads, 1);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads_kv, 1);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads_kv, 1);
    ggml_tensor * cache = ggml_new_tensor_4d(ctx, GGML_TYPE_Q8_0, head_dim, block_size, 2 * n_heads_kv, n_blocks);
    ggml_tensor * block_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, max_blocks, 1);
    ggml_tensor * write_slots = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * context_lens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * batch_offsets = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * batch_lens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * output = ggml_qwen38_paged_attn(
        ctx, q, k, v, cache, block_table, write_slots, context_lens, batch_offsets, batch_lens,
        1.0f / std::sqrt(static_cast<float>(head_dim)), block_size, max_blocks, 4);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        fprintf(stderr, "%s : failed to allocate test tensors\n", __func__);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }
    ggml_backend_buffer_clear(buffer, 0);

    std::vector<float> q_data(ggml_nelements(q), 0.0f);
    std::vector<float> k_data(ggml_nelements(k));
    std::vector<float> v_data(ggml_nelements(v));
    for (size_t i = 0; i < k_data.size(); ++i) {
        k_data[i] = static_cast<float>(static_cast<int>(i % 31) - 15) / 31.0f;
        v_data[i] = static_cast<float>(static_cast<int>(i % 127) - 63) / 63.0f;
    }
    const int32_t block_table_data[max_blocks] = { 4, 3, 2, 1, 0 };
    const int32_t write_slot_data[1] = { 0 };
    const int32_t context_len_data[1] = { 65 };
    const int32_t batch_offset_data[1] = { 0 };
    const int32_t batch_len_data[1] = { 1 };

    ggml_backend_tensor_set(q, q_data.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, k_data.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, v_data.data(), 0, ggml_nbytes(v));
    ggml_backend_tensor_set(block_table, block_table_data, 0, sizeof(block_table_data));
    ggml_backend_tensor_set(write_slots, write_slot_data, 0, sizeof(write_slot_data));
    ggml_backend_tensor_set(context_lens, context_len_data, 0, sizeof(context_len_data));
    ggml_backend_tensor_set(batch_offsets, batch_offset_data, 0, sizeof(batch_offset_data));
    ggml_backend_tensor_set(batch_lens, batch_len_data, 0, sizeof(batch_len_data));

    std::vector<uint8_t> cache_data(ggml_nbytes(cache), 0);
    for (int physical_page = 1; physical_page <= 4; ++physical_page) {
        for (int head = 0; head < n_heads_kv; ++head) {
            for (int token = 0; token < block_size; ++token) {
                const size_t k_offset = static_cast<size_t>(physical_page) * cache->nb[3] + static_cast<size_t>(head) * cache->nb[2] + static_cast<size_t>(token) * cache->nb[1];
                const size_t v_offset = static_cast<size_t>(physical_page) * cache->nb[3] + static_cast<size_t>(n_heads_kv + head) * cache->nb[2] + static_cast<size_t>(token) * cache->nb[1];
                ggml_quantize_chunk(GGML_TYPE_Q8_0, k_data.data() + head * head_dim, cache_data.data() + k_offset, 0, 1, head_dim, nullptr);
                ggml_quantize_chunk(GGML_TYPE_Q8_0, v_data.data() + head * head_dim, cache_data.data() + v_offset, 0, 1, head_dim, nullptr);
            }
        }
    }
    ggml_backend_tensor_set(cache, cache_data.data(), 0, cache_data.size());

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, output);
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s : graph compute failed: %s\n", __func__, ggml_status_to_string(status));
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    std::vector<float> output_data(ggml_nelements(output));
    ggml_backend_tensor_get(output, output_data.data(), 0, ggml_nbytes(output));
    std::vector<uint8_t> physical_page_0(cache->nb[3]);
    std::vector<uint8_t> physical_page_1(cache->nb[3]);
    std::vector<uint8_t> physical_page_5(cache->nb[3]);
    ggml_backend_tensor_get(cache, physical_page_0.data(), 0, physical_page_0.size());
    ggml_backend_tensor_get(cache, physical_page_1.data(), cache->nb[3], physical_page_1.size());
    ggml_backend_tensor_get(cache, physical_page_5.data(), 5 * cache->nb[3], physical_page_5.size());
    for (uint8_t value : physical_page_5) {
        if (value != 0) {
            fprintf(stderr, "%s : write escaped into an unreferenced physical page\n", __func__);
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            ggml_backend_free(backend);
            return 1;
        }
    }
    bool referenced_page_written = false;
    for (uint8_t value : physical_page_0) {
        referenced_page_written |= value != 0;
    }
    for (uint8_t value : physical_page_1) {
        referenced_page_written |= value != 0;
    }
    if (!referenced_page_written) {
        fprintf(stderr, "%s : referenced physical page was not written\n", __func__);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }
    for (int head = 0; head < n_heads; ++head) {
        const int kv_head = head / (n_heads / n_heads_kv);
        for (int dim = 0; dim < head_dim; ++dim) {
            const float expected = v_data[kv_head * head_dim + dim];
            const float actual = output_data[head * head_dim + dim];
            if (std::fabs(actual - expected) > 0.01f) {
                fprintf(stderr, "%s : output mismatch at head %d dim %d: %g != %g\n", __func__, head, dim, actual, expected);
                ggml_backend_buffer_free(buffer);
                ggml_free(ctx);
                ggml_backend_free(backend);
                return 1;
            }
        }
    }

    fprintf(stderr, "%s : Q8 paged attention CUDA output matches quantized reference\n", __func__);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
