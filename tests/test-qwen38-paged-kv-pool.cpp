#include "arg.h"
#include "common.h"
#include "src/llama-model.h"
#include "src/llama-paged-kv-pool.h"

#include <clocale>
#include <cstdio>

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    ggml_backend_load_all();
    common_init_result_ptr init = common_init_from_params(params, true);
    llama_model * model = init->model();
    if (model == nullptr || model->arch != LLM_ARCH_QWEN35 || !llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s : Qwen3.8 hybrid model is required\n", __func__);
        return 1;
    }

    const auto filter = [&](int32_t il) {
        return il < static_cast<int32_t>(model->hparams.n_layer()) && !model->hparams.is_recr(il);
    };
    llama_paged_kv_pool pool(*model, GGML_TYPE_Q8_0, 16, 64, true, filter);
    if (pool.n_devices() != 2 || pool.page_size() != 16 || pool.n_pages() != 64) {
        fprintf(stderr, "%s : invalid paged pool topology\n", __func__);
        return 1;
    }
    const bool tensor_sharded = model->split_mode() == LLAMA_SPLIT_MODE_TENSOR;
    if (pool.tensor_sharded() != tensor_sharded) {
        fprintf(stderr, "%s : paged pool split mode mismatch\n", __func__);
        return 1;
    }

    int attention_layers = 0;
    for (int32_t il = 0; il < static_cast<int32_t>(model->hparams.n_layer()); ++il) {
        ggml_tensor * kv = pool.get_kv(il);
        if (model->hparams.is_recr(il)) {
            if (kv != nullptr) {
                fprintf(stderr, "%s : recurrent layer %d owns paged KV\n", __func__, il);
                return 1;
            }
            continue;
        }

        ++attention_layers;
        if (kv == nullptr || kv->type != GGML_TYPE_Q8_0 || kv->ne[0] != 256 || kv->ne[1] != 16 || kv->ne[2] != 64 || kv->ne[3] != 8) {
            fprintf(stderr, "%s : invalid paged tensor for layer %d\n", __func__, il);
            return 1;
        }
        const auto buft = ggml_backend_buffer_get_type(kv->buffer);
        if (ggml_backend_buft_get_device(buft) != model->dev_layer(il)) {
            fprintf(stderr, "%s : paged tensor for layer %d is on the wrong device\n", __func__, il);
            return 1;
        }
        const uint32_t expected_device = tensor_sharded ? 0 : (il < 36 ? 0 : 1);
        if (pool.device_index(il) != expected_device) {
            fprintf(stderr, "%s : layer %d has device index %u instead of %u\n", __func__, il, pool.device_index(il), expected_device);
            return 1;
        }
    }
    if (attention_layers != 16) {
        fprintf(stderr, "%s : expected 16 attention layers, got %d\n", __func__, attention_layers);
        return 1;
    }

    fprintf(stderr, "%s : 16 Q8 paged KV layers use %s placement\n", __func__, tensor_sharded ? "tensor-sharded" : "36/28 layer");
    return 0;
}
