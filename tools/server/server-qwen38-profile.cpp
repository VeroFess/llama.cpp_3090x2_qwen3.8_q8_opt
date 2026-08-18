#include "server-qwen38-profile.h"

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "src/llama-model.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>
#include <vector>

#if defined(GGML_QWEN38_2X3090)
#include <openssl/evp.h>
#endif

static constexpr const char * QWEN38_PROFILE = "qwen38-27b-q8-2x3090";
static constexpr int32_t QWEN38_CONTEXT = 262144;
static constexpr int32_t QWEN38_SEQUENCES = 8;

struct qwen38_tuning_config {
    int32_t layer_boundary = 36;
    int32_t max_num_batched_tokens = 2048;
    int32_t prefill_chunk_size = 2048;
    int32_t ubatch_size = 64;
    int32_t kv_page_size = 16;
    float scheduler_target_step_ms = 10.0f;
    int32_t mtp_n_max = 2;
};

#if defined(GGML_QWEN38_2X3090)
static std::string qwen38_sha256(const void * data, size_t size, std::string & error) {
    EVP_MD_CTX * context = EVP_MD_CTX_new();
    if (context == nullptr) {
        error = "failed to allocate SHA-256 context";
        return {};
    }
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_size = 0;
    const bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
        EVP_DigestUpdate(context, data, size) == 1 &&
        EVP_DigestFinal_ex(context, digest, &digest_size) == 1;
    EVP_MD_CTX_free(context);
    if (!ok) {
        error = "failed to compute SHA-256";
        return {};
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < digest_size; ++i) {
        output << std::setw(2) << static_cast<unsigned int>(digest[i]);
    }
    return output.str();
}

static bool qwen38_nonempty_string(const nlohmann::json & object, const char * key) {
    return object.contains(key) && object.at(key).is_string() && !object.at(key).get<std::string>().empty();
}

static bool qwen38_load_tuning(common_params & params, qwen38_tuning_config & config, std::string & error) {
    if (params.autotune_mode == "off") {
        if (!params.autotune_cache_path.empty()) {
            error = "--autotune-cache requires --autotune quick";
            return false;
        }
        return true;
    }
    if (params.autotune_mode != "quick") {
        error = "unsupported autotune mode: " + params.autotune_mode;
        return false;
    }
    if (params.autotune_cache_path.empty()) {
        error = "--autotune quick requires --autotune-cache";
        return false;
    }

    std::ifstream input(params.autotune_cache_path);
    if (!input) {
        error = "failed to open autotune cache: " + params.autotune_cache_path;
        return false;
    }

    nlohmann::json root;
    try {
        input >> root;
    } catch (const std::exception & exception) {
        error = "failed to parse autotune cache: " + std::string(exception.what());
        return false;
    }

    try {
        if (root.at("schema_version").get<int>() != 2 || root.at("profile").get<std::string>() != QWEN38_PROFILE) {
            error = "autotune cache schema or profile mismatch";
            return false;
        }
        if (!root.at("validated").get<bool>()) {
            error = "autotune cache has not passed its measurement suite";
            return false;
        }
        const auto & key = root.at("key");
        for (const char * field : {"model_sha256", "gguf_inventory_sha256", "llama_commit", "fork_patch_version",
                "cuda_toolkit", "driver_version", "pcie_topology_sha256", "kv_codec"}) {
            if (!qwen38_nonempty_string(key, field)) {
                error = std::string("autotune cache key is missing ") + field;
                return false;
            }
        }
        if (!key.at("gpu_uuids").is_array() || key.at("gpu_uuids").size() != 2 ||
                !key.at("power_limits_w").is_array() || key.at("power_limits_w").size() != 2 ||
                key.at("kv_codec").get<std::string>() != "q8_0" ||
                key.at("max_resident_sequences").get<int>() != QWEN38_SEQUENCES) {
            error = "autotune cache hardware or quality key mismatch";
            return false;
        }
        const std::string canonical_key = key.dump();
        const std::string computed_key_sha256 = qwen38_sha256(canonical_key.data(), canonical_key.size(), error);
        if (computed_key_sha256.empty() || root.at("key_sha256").get<std::string>() != computed_key_sha256) {
            error = "autotune cache key SHA-256 mismatch";
            return false;
        }

        const auto & selected = root.at("selected");
        config.layer_boundary = selected.at("layer_boundary").get<int32_t>();
        config.max_num_batched_tokens = selected.at("max_num_batched_tokens").get<int32_t>();
        config.prefill_chunk_size = selected.at("prefill_chunk_size").get<int32_t>();
        config.ubatch_size = selected.at("ubatch_size").get<int32_t>();
        config.kv_page_size = selected.at("kv_page_size").get<int32_t>();
        config.scheduler_target_step_ms = selected.at("scheduler_target_step_ms").get<float>();
        config.mtp_n_max = selected.at("mtp_n_max").get<int32_t>();
        if ((config.layer_boundary != 28 && config.layer_boundary != 32 && config.layer_boundary != 36) ||
                (config.max_num_batched_tokens != 512 && config.max_num_batched_tokens != 1024 && config.max_num_batched_tokens != 2048) ||
                config.prefill_chunk_size != config.max_num_batched_tokens ||
                (config.ubatch_size != 16 && config.ubatch_size != 32 && config.ubatch_size != 64) ||
                config.ubatch_size > config.max_num_batched_tokens ||
                (config.kv_page_size != 16 && config.kv_page_size != 32) ||
                config.scheduler_target_step_ms <= 0.0f || config.scheduler_target_step_ms > 50.0f ||
                (config.mtp_n_max != 1 && config.mtp_n_max != 2) ||
                selected.at("pipeline_depth").get<int>() != 2 ||
                selected.at("activation_buffer_count").get<int>() != 3) {
            error = "autotune cache selected configuration is outside safe profile bounds";
            return false;
        }
        if ((params.max_num_batched_tokens != 0 && params.max_num_batched_tokens != config.max_num_batched_tokens) ||
                (params.kv_page_size != 0 && params.kv_page_size != config.kv_page_size) ||
                (params.scheduler_target_step_ms != 0.0f && params.scheduler_target_step_ms != config.scheduler_target_step_ms)) {
            error = "explicit runtime option conflicts with autotune cache";
            return false;
        }

        params.autotune_cache_key_sha256 = computed_key_sha256;
        params.autotune_model_sha256 = key.at("model_sha256").get<std::string>();
    } catch (const std::exception & exception) {
        error = "invalid autotune cache: " + std::string(exception.what());
        return false;
    }
    return true;
}
#endif

bool server_qwen38_profile_enabled(const common_params & params) {
    return params.deployment_profile == QWEN38_PROFILE;
}

static bool has_tensor_split(const common_params & params) {
    return std::any_of(std::begin(params.tensor_split), std::end(params.tensor_split), [](float value) {
        return value != 0.0f;
    });
}

bool server_qwen38_profile_apply(common_params & params, std::string & error) {
    if (params.deployment_profile.empty()) {
        return true;
    }
    if (!server_qwen38_profile_enabled(params)) {
        error = "unknown deployment profile: " + params.deployment_profile;
        return false;
    }
    if (params.deployment_profile_applied) {
        return true;
    }

#if !defined(GGML_QWEN38_2X3090)
    error = "the qwen38-27b-q8-2x3090 profile is not compiled into this binary";
    return false;
#else
    qwen38_tuning_config tuning;
    if (!qwen38_load_tuning(params, tuning, error)) {
        return false;
    }
    const bool tensor_mode = params.split_mode == LLAMA_SPLIT_MODE_TENSOR;
    if (params.split_mode != LLAMA_SPLIT_MODE_LAYER && !tensor_mode) {
        error = "the Qwen3.8 profile requires --split-mode layer or tensor";
        return false;
    }
    if (tensor_mode && params.autotune_mode != "off") {
        error = "tensor mode does not use the layer-placement autotune cache";
        return false;
    }
    if (tensor_mode && has_tensor_split(params)) {
        const bool equal_split = params.tensor_split[0] > 0.0f && params.tensor_split[1] > 0.0f &&
            std::fabs(params.tensor_split[0] - params.tensor_split[1]) < 1e-6f &&
            std::all_of(std::begin(params.tensor_split) + 2, std::end(params.tensor_split), [](float value) { return value == 0.0f; });
        if (!equal_split) {
            error = "the Qwen3.8 tensor profile requires an equal 1:1 tensor split";
            return false;
        }
    } else if (!tensor_mode && has_tensor_split(params)) {
        error = "the Qwen3.8 profile rejects --tensor-split";
        return false;
    }
    if (params.n_ctx != 0 && params.n_ctx != QWEN38_CONTEXT) {
        error = "the Qwen3.8 profile requires --max-model-len 262144";
        return false;
    }
    if (params.n_parallel != 1 && params.n_parallel != -1 && params.n_parallel != QWEN38_SEQUENCES) {
        error = "the Qwen3.8 profile requires --max-num-seqs 8";
        return false;
    }
    if (params.n_gpu_layers != -1 && params.n_gpu_layers != -2) {
        error = "the Qwen3.8 profile requires full GPU offload";
        return false;
    }
    if (params.cache_type_k != GGML_TYPE_F16 && params.cache_type_k != GGML_TYPE_Q8_0) {
        error = "the Qwen3.8 quality profile requires q8_0 K cache";
        return false;
    }
    if (params.cache_type_v != GGML_TYPE_F16 && params.cache_type_v != GGML_TYPE_Q8_0) {
        error = "the Qwen3.8 quality profile requires q8_0 V cache";
        return false;
    }
    if (params.ctx_shift) {
        error = "the Qwen3.8 profile rejects context shift";
        return false;
    }
    if (params.no_kv_offload) {
        error = "the Qwen3.8 profile rejects CPU KV storage";
        return false;
    }
    if (!params.mmproj.empty() || !params.image.empty()) {
        error = "the Qwen3.8 profile does not support image or video input";
        return false;
    }
    if (params.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN || params.yarn_orig_ctx != 0 || params.yarn_ext_factor >= 0.0f) {
        error = "the Qwen3.8 profile rejects YaRN";
        return false;
    }

    params.n_ctx = QWEN38_CONTEXT;
    params.n_parallel = QWEN38_SEQUENCES;
    params.n_gpu_layers = -2;
    params.split_mode = tensor_mode ? LLAMA_SPLIT_MODE_TENSOR : LLAMA_SPLIT_MODE_LAYER;
    if (tensor_mode) {
        params.tensor_split[0] = 1.0f;
        params.tensor_split[1] = 1.0f;
        params.qwen38_layer_boundary = -1;
    } else {
        params.tensor_split[0] = tuning.layer_boundary / 66.0f;
        params.tensor_split[1] = (66 - tuning.layer_boundary) / 66.0f;
        params.qwen38_layer_boundary = tuning.layer_boundary;
    }
    params.fit_params = false;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.cache_type_k = GGML_TYPE_Q8_0;
    params.cache_type_v = GGML_TYPE_Q8_0;
    params.ctx_shift = false;
    params.no_kv_offload = false;
    params.no_mmproj = true;
    params.kv_unified = true;
    params.kv_paged_storage = true;
    params.cont_batching = true;
    params.n_ctx_checkpoints = 2;
    params.checkpoint_min_step = 32768;
    params.cache_ram_mib = std::min(params.cache_ram_mib, 2048);
    params.max_num_batched_tokens = params.autotune_mode == "quick" ? tuning.max_num_batched_tokens :
        (params.max_num_batched_tokens == 0 ? 2048 : params.max_num_batched_tokens);
    params.n_batch = params.max_num_batched_tokens;
    params.n_ubatch = params.autotune_mode == "quick" ? tuning.ubatch_size : std::min(params.n_ubatch, 1024);
    params.gpu_memory_reserve_mib = params.gpu_memory_reserve_mib == 0 ? 1024 : params.gpu_memory_reserve_mib;
    params.kv_page_size = params.autotune_mode == "quick" ? tuning.kv_page_size : (params.kv_page_size == 0 ? 16 : params.kv_page_size);
    params.scheduler_target_step_ms = params.autotune_mode == "quick" ? tuning.scheduler_target_step_ms :
        (params.scheduler_target_step_ms == 0.0f ? 10.0f : params.scheduler_target_step_ms);
    params.sampling.backend_sampling = true;
    params.endpoint_metrics = true;

    auto & types = params.speculative.types;
    types.erase(std::remove(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_NONE), types.end());
    if (std::find(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) == types.end()) {
        types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
    }
    params.speculative.draft.cache_type_k = GGML_TYPE_Q8_0;
    params.speculative.draft.cache_type_v = GGML_TYPE_Q8_0;
    params.speculative.draft.backend_sampling = true;
    params.speculative.draft.n_max = params.autotune_mode == "quick" ? tuning.mtp_n_max : 2;
    params.deployment_profile_applied = true;

    return true;
#endif
}

bool server_qwen38_profile_validate_hardware(common_params & params, std::string & error) {
    if (!server_qwen38_profile_enabled(params)) {
        return true;
    }

#if !defined(GGML_QWEN38_2X3090) || !defined(GGML_USE_CUDA)
    error = "the Qwen3.8 profile requires the CUDA target build";
    return false;
#else
    const int device_count = ggml_backend_cuda_get_device_count();
    if (device_count != 2) {
        error = "the Qwen3.8 profile requires exactly two visible CUDA devices";
        return false;
    }

    constexpr size_t min_vram = 23ull * 1024 * 1024 * 1024;
    constexpr size_t max_vram = 25ull * 1024 * 1024 * 1024;
    for (int device = 0; device < device_count; ++device) {
        std::array<char, 256> description{};
        size_t free = 0;
        size_t total = 0;
        ggml_backend_cuda_get_device_description(device, description.data(), description.size());
        ggml_backend_cuda_get_device_memory(device, &free, &total);
        if (std::strstr(description.data(), "RTX 3090") == nullptr) {
            error = "CUDA device " + std::to_string(device) + " is not an RTX 3090: " + description.data();
            return false;
        }
        if (total < min_vram || total > max_vram) {
            error = "CUDA device " + std::to_string(device) + " does not have 24 GB class VRAM";
            return false;
        }
        if (ggml_backend_cuda_get_device_compute_capability(device) != 860) {
            error = "CUDA device " + std::to_string(device) + " is not SM86";
            return false;
        }
    }

    const bool can_access_peer = ggml_backend_cuda_can_access_peer(0, 1) && ggml_backend_cuda_can_access_peer(1, 0);
    const bool peer_enabled = can_access_peer && ggml_backend_cuda_enable_peer_access(0, 1) && ggml_backend_cuda_enable_peer_access(1, 0);
    params.cuda_p2p_active = peer_enabled;
    if (peer_enabled) {
        LOG_INF("qwen38 profile: hardware validated, 2x RTX 3090, SM86, bidirectional CUDA P2P\n");
    } else {
        LOG_INF("qwen38 profile: hardware validated, 2x RTX 3090, SM86, pinned host-staged inter-stage transfers\n");
    }
    return true;
#endif
}

static std::string qwen38_model_sha256(const std::string & path, std::string & error) {
#if !defined(GGML_QWEN38_2X3090)
    (void) path;
    error = "SHA-256 is unavailable in a generic build";
    return {};
#else
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "failed to open the model for SHA-256: " + path;
        return {};
    }

    EVP_MD_CTX * context = EVP_MD_CTX_new();
    if (context == nullptr || EVP_DigestInit_ex(context, EVP_sha256(), nullptr) != 1) {
        EVP_MD_CTX_free(context);
        error = "failed to initialize SHA-256";
        return {};
    }

    std::vector<char> buffer(16 * 1024 * 1024);
    while (input) {
        input.read(buffer.data(), buffer.size());
        const std::streamsize count = input.gcount();
        if (count > 0 && EVP_DigestUpdate(context, buffer.data(), static_cast<size_t>(count)) != 1) {
            EVP_MD_CTX_free(context);
            error = "failed while hashing the model";
            return {};
        }
    }

    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int digest_size = 0;
    if (EVP_DigestFinal_ex(context, digest.data(), &digest_size) != 1) {
        EVP_MD_CTX_free(context);
        error = "failed to finalize the model hash";
        return {};
    }
    EVP_MD_CTX_free(context);

    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < digest_size; ++i) {
        output << std::setw(2) << static_cast<unsigned int>(digest[i]);
    }
    return output.str();
#endif
}

bool server_qwen38_profile_validate_model(common_params & params, const llama_model * model_ptr, const llama_context * ctx, std::string & error) {
    if (!server_qwen38_profile_enabled(params)) {
        return true;
    }
    if (model_ptr == nullptr || ctx == nullptr) {
        error = "the Qwen3.8 profile received an incomplete model context";
        return false;
    }

    const auto * model = static_cast<const llama_model *>(model_ptr);
    const auto & hparams = model->hparams;
    const bool tensor_mode = model->split_mode() == LLAMA_SPLIT_MODE_TENSOR;
    if (model->arch != LLM_ARCH_QWEN35 || (model->split_mode() != LLAMA_SPLIT_MODE_LAYER && !tensor_mode)) {
        error = "the model is not a supported Qwen3.8 multi-GPU GGUF";
        return false;
    }
    if (hparams.n_layer() != 64 || hparams.n_embd != 5120 || hparams.n_ctx_train != QWEN38_CONTEXT) {
        error = "the model does not match Qwen3.8-27B dimensions";
        return false;
    }
    if (hparams.n_head() != 24 || hparams.n_head_kv() != 4 || hparams.n_embd_head_k() != 256 || hparams.n_rot() != 64) {
        error = "the model does not match the Qwen3.8 attention layout";
        return false;
    }

    int recurrent_layers = 0;
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        recurrent_layers += hparams.is_recr(il) ? 1 : 0;
    }
    if (recurrent_layers != 48 || hparams.n_layer_nextn < 1) {
        error = "the model does not contain the required DeltaNet and MTP layout";
        return false;
    }

    int layer_boundary = -1;
    if (tensor_mode) {
        ggml_backend_dev_t meta_device = model->dev_layer(0);
        if (ggml_backend_dev_type(meta_device) != GGML_BACKEND_DEVICE_TYPE_META) {
            error = "tensor mode did not create a Meta device";
            return false;
        }
        for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
            if (model->dev_layer(il) != meta_device) {
                error = "tensor mode placed layer " + std::to_string(il) + " outside the Meta device";
                return false;
            }
        }
        if (model->dev_output() != meta_device) {
            error = "tensor mode placed the output head outside the Meta device";
            return false;
        }
        params.qwen38_layer_boundary = -1;
    } else {
        ggml_backend_dev_t first_device = model->dev_layer(0);
        ggml_backend_dev_t second_device = nullptr;
        for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
            if (ggml_backend_dev_type(model->dev_layer(il)) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                error = "model layer " + std::to_string(il) + " is not on a GPU";
                return false;
            }
            if (model->dev_layer(il) != first_device && layer_boundary < 0) {
                layer_boundary = static_cast<int>(il);
                second_device = model->dev_layer(il);
            }
            if (layer_boundary >= 0 && model->dev_layer(il) == first_device) {
                error = "model layers are not assigned in two contiguous ranges";
                return false;
            }
        }
        if (layer_boundary < 28 || layer_boundary > 36 || layer_boundary % 4 != 0) {
            error = "the selected layer boundary " + std::to_string(layer_boundary) + " is outside the Qwen3.8 group candidates";
            return false;
        }
        params.qwen38_layer_boundary = layer_boundary;
        for (uint32_t il = hparams.n_layer(); il < hparams.n_layer_all; ++il) {
            if (model->dev_layer(il) != second_device) {
                error = "MTP layer " + std::to_string(il) + " is not on the second pipeline stage";
                return false;
            }
        }
        if (model->dev_output() != second_device) {
            error = "the output head is not on the second pipeline stage";
            return false;
        }
    }

    size_t q8_matrices = 0;
    size_t rejected_matrices = 0;
    for (const auto & entry : model->tensors_by_name) {
        const ggml_tensor * tensor = entry.second;
        if (tensor == nullptr || ggml_n_dims(tensor) < 2 || ggml_nelements(tensor) < 1024 * 1024) {
            continue;
        }
        if (tensor->type == GGML_TYPE_Q8_0) {
            ++q8_matrices;
        } else {
            ++rejected_matrices;
        }
    }
    if (q8_matrices == 0 || rejected_matrices != 0) {
        error = "the model contains a core matrix below the Q8 quality floor";
        return false;
    }
    if (llama_n_ctx(ctx) != QWEN38_CONTEXT || llama_n_seq_max(ctx) != QWEN38_SEQUENCES) {
        error = "the runtime context does not match the profile limits";
        return false;
    }

    params.model_sha256 = qwen38_model_sha256(params.model.path, error);
    if (params.model_sha256.empty()) {
        return false;
    }
    if (params.autotune_mode == "quick" && params.model_sha256 != params.autotune_model_sha256) {
        error = "autotune cache model SHA-256 does not match the loaded model";
        return false;
    }

    LOG_INF("qwen38 profile: model_sha256=%s\n", params.model_sha256.c_str());
    if (params.autotune_mode == "quick") {
        LOG_INF("qwen38 profile: autotune cache validated, key_sha256=%s\n", params.autotune_cache_key_sha256.c_str());
    }
    if (tensor_mode) {
        LOG_INF("qwen38 profile: model validated, layers=64, tensor_split=1/1, recurrent=48, attention=16, q8_matrices=%zu, mtp_layers=%u\n",
            q8_matrices, hparams.n_layer_nextn);
    } else {
        LOG_INF("qwen38 profile: model validated, layers=64, boundary=%d/%d, recurrent=48, attention=16, q8_matrices=%zu, mtp_layers=%u\n",
            layer_boundary, 64 - layer_boundary, q8_matrices, hparams.n_layer_nextn);
    }
    LOG_INF("qwen38 profile: max_model_len=%d, max_num_seqs=%d, kv=q8_0, page_size=%d, reserve_mib=%d\n",
        QWEN38_CONTEXT, QWEN38_SEQUENCES, params.kv_page_size, params.gpu_memory_reserve_mib);
    return true;
}
