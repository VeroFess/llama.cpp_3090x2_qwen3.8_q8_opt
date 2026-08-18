#include "server-qwen38-profile.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

static void require_impl(bool condition, const char * expression, int line) {
    if (!condition) {
        std::fprintf(stderr, "require failed at line %d: %s\n", line, expression);
        std::abort();
    }
}

#define require(condition) require_impl((condition), #condition, __LINE__)

static common_params profile_params() {
    common_params params;
    params.deployment_profile = "qwen38-27b-q8-2x3090";
    return params;
}

static nlohmann::json tuning_cache() {
    return {
        {"schema_version", 2},
        {"profile", "qwen38-27b-q8-2x3090"},
        {"validated", true},
        {"key", {
            {"model_sha256", std::string(64, 'a')},
            {"gguf_inventory_sha256", std::string(64, 'b')},
            {"llama_commit", "dc57e5ecb4d4"},
            {"fork_patch_version", "qwen38-v1"},
            {"cuda_toolkit", "13.0"},
            {"driver_version", "580.95"},
            {"gpu_uuids", {"GPU-0", "GPU-1"}},
            {"power_limits_w", {350.0, 350.0}},
            {"pcie_topology_sha256", std::string(64, 'c')},
            {"kv_codec", "q8_0"},
            {"max_resident_sequences", 8},
        }},
        {"key_sha256", "805ae5eac0ccbb8041bab5fb0cff24c8d371e871f5260c2cf69b1b6e44a4d69e"},
        {"selected", {
            {"layer_boundary", 32},
            {"max_num_batched_tokens", 1024},
            {"prefill_chunk_size", 1024},
            {"ubatch_size", 32},
            {"kv_page_size", 32},
            {"scheduler_target_step_ms", 20.0},
            {"mtp_n_max", 1},
            {"pipeline_depth", 2},
            {"activation_buffer_count", 3},
        }},
    };
}

static std::filesystem::path write_tuning_cache(const nlohmann::json & cache, const std::string & name) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream output(path);
    output << cache.dump(2);
    output.close();
    require(output.good());
    return path;
}

static void test_defaults() {
    common_params params = profile_params();
    std::string error;
    const bool applied = server_qwen38_profile_apply(params, error);
    if (!applied) {
        std::fprintf(stderr, "profile error: %s\n", error.c_str());
    }
    require(applied);
    require(server_qwen38_profile_apply(params, error));
    require(params.n_ctx == 262144);
    require(params.n_parallel == 8);
    require(params.n_gpu_layers == -2);
    require(params.split_mode == LLAMA_SPLIT_MODE_LAYER);
    require(params.tensor_split[0] == 36.0f / 66.0f);
    require(params.tensor_split[1] == 30.0f / 66.0f);
    require(params.qwen38_layer_boundary == 36);
    require(!params.fit_params);
    require(params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_ENABLED);
    require(params.cache_type_k == GGML_TYPE_Q8_0);
    require(params.cache_type_v == GGML_TYPE_Q8_0);
    require(params.kv_unified);
    require(params.kv_paged_storage);
    require(params.n_ctx_checkpoints == 2);
    require(params.speculative.draft.n_max == 2);
    require(params.speculative.draft.cache_type_k == GGML_TYPE_Q8_0);
    require(params.speculative.draft.cache_type_v == GGML_TYPE_Q8_0);
    require(!params.cuda_p2p_active);
    require(std::find(params.speculative.types.begin(), params.speculative.types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end());

    const auto cparams = common_context_params_to_llama(params);
    require(cparams.kv_page_size == 16);
}

static void test_rejections() {
    std::string error;

    common_params row = profile_params();
    row.split_mode = LLAMA_SPLIT_MODE_ROW;
    require(!server_qwen38_profile_apply(row, error));

    common_params short_ctx = profile_params();
    short_ctx.n_ctx = 131072;
    require(!server_qwen38_profile_apply(short_ctx, error));

    common_params low_kv = profile_params();
    low_kv.cache_type_k = GGML_TYPE_Q4_0;
    require(!server_qwen38_profile_apply(low_kv, error));

    common_params shifted = profile_params();
    shifted.ctx_shift = true;
    require(!server_qwen38_profile_apply(shifted, error));

    common_params yarn = profile_params();
    yarn.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
    require(!server_qwen38_profile_apply(yarn, error));
}

static void test_autotune_cache() {
    std::string error;
    common_params missing = profile_params();
    missing.autotune_mode = "quick";
    require(!server_qwen38_profile_apply(missing, error));

    const auto valid_path = write_tuning_cache(tuning_cache(), "qwen38-valid-tuning.json");
    common_params tuned = profile_params();
    tuned.autotune_mode = "quick";
    tuned.autotune_cache_path = valid_path.string();
    require(server_qwen38_profile_apply(tuned, error));
    require(tuned.qwen38_layer_boundary == 32);
    require(tuned.tensor_split[0] == 32.0f / 66.0f);
    require(tuned.tensor_split[1] == 34.0f / 66.0f);
    require(tuned.max_num_batched_tokens == 1024);
    require(tuned.n_ubatch == 32);
    require(tuned.kv_page_size == 32);
    require(tuned.scheduler_target_step_ms == 20.0f);
    require(tuned.speculative.draft.n_max == 1);
    require(tuned.autotune_model_sha256 == std::string(64, 'a'));
    require(!tuned.autotune_cache_key_sha256.empty());
    std::filesystem::remove(valid_path);

    auto invalid_sha = tuning_cache();
    invalid_sha["key_sha256"] = std::string(64, '0');
    const auto invalid_sha_path = write_tuning_cache(invalid_sha, "qwen38-invalid-sha-tuning.json");
    common_params bad_sha = profile_params();
    bad_sha.autotune_mode = "quick";
    bad_sha.autotune_cache_path = invalid_sha_path.string();
    require(!server_qwen38_profile_apply(bad_sha, error));
    std::filesystem::remove(invalid_sha_path);

    auto unsafe = tuning_cache();
    unsafe["selected"]["layer_boundary"] = 30;
    const auto unsafe_path = write_tuning_cache(unsafe, "qwen38-unsafe-tuning.json");
    common_params bad_config = profile_params();
    bad_config.autotune_mode = "quick";
    bad_config.autotune_cache_path = unsafe_path.string();
    require(!server_qwen38_profile_apply(bad_config, error));
    std::filesystem::remove(unsafe_path);
}

int main() {
#if defined(GGML_QWEN38_2X3090)
    test_defaults();
    test_rejections();
    test_autotune_cache();
#else
    common_params params = profile_params();
    std::string error;
    require(!server_qwen38_profile_apply(params, error));
#endif
    return 0;
}
