#include "arg.h"
#include "common.h"
#include "llama.h"
#include "src/llama-memory-hybrid.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <vector>

static bool decode_one(
        llama_context * ctx,
        llama_token token,
        llama_pos pos,
        llama_seq_id sequence,
        int n_vocab,
        std::vector<float> & logits) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, token, pos, { sequence }, true);
    const bool decoded = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    if (!decoded) {
        return false;
    }
    const float * values = llama_get_logits_ith(ctx, -1);
    if (values == nullptr) {
        return false;
    }
    logits.assign(values, values + n_vocab);
    return true;
}

static int argmax(const std::vector<float> & values) {
    return std::max_element(values.begin(), values.end()) - values.begin();
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.n_ctx = 64;

    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();
    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to load model\n", __func__);
        return 1;
    }
    if (!llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s : skipping non-hybrid model\n", __func__);
        return 0;
    }

    auto cparams = common_context_params_to_llama(params);
    cparams.n_ctx = 64;
    cparams.n_seq_max = 2;
    cparams.n_rs_seq = 8;
    cparams.kv_page_size = 16;
    cparams.kv_paged_storage = true;
    cparams.n_batch = std::max(cparams.n_batch, 10u);
    cparams.n_ubatch = std::max(cparams.n_ubatch, 10u);

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "%s : failed to create context\n", __func__);
        return 1;
    }

    llama_memory_t mem = llama_get_memory(ctx);
    if (!llama_memory_paged_admit(mem, 0, 32)) {
        fprintf(stderr, "%s : paged admission failed\n", __func__);
        llama_free(ctx);
        return 1;
    }
    llama_paged_memory_stats paged_stats{};
    if (!llama_memory_paged_get_stats(mem, &paged_stats) || paged_stats.admitted_pages != 2 || paged_stats.committed_pages != 2 || paged_stats.free_pages != paged_stats.total_pages) {
        fprintf(stderr, "%s : invalid admission statistics\n", __func__);
        llama_free(ctx);
        return 1;
    }

    std::vector<llama_token> tokens = common_tokenize(ctx, "The quick brown fox jumps over the lazy dog", true);
    if (tokens.empty()) {
        fprintf(stderr, "%s : tokenization failed\n", __func__);
        llama_free(ctx);
        return 1;
    }
    tokens.resize(17, tokens.back());

    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (uint32_t pos = 0; pos < tokens.size(); ++pos) {
        common_batch_add(batch, tokens[pos], pos, { 0 }, pos + 1 == tokens.size());
    }
    const bool decode_ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    if (!decode_ok) {
        fprintf(stderr, "%s : decode failed\n", __func__);
        llama_free(ctx);
        return 1;
    }

    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem);
    if (hybrid == nullptr || hybrid->get_block_manager() == nullptr) {
        fprintf(stderr, "%s : hybrid paged block manager is not active\n", __func__);
        llama_free(ctx);
        return 1;
    }
    const size_t decoded_pages = (tokens.size() + cparams.kv_page_size - 1) / cparams.kv_page_size;
    if (!llama_memory_paged_get_stats(mem, &paged_stats) || paged_stats.admitted_pages != 2 || paged_stats.committed_pages != 2 || paged_stats.free_pages + decoded_pages != paged_stats.total_pages) {
        fprintf(stderr, "%s : invalid committed-page statistics\n", __func__);
        llama_free(ctx);
        return 1;
    }
    llama_memory_paged_release_admission(mem, 0);
    if (!llama_memory_paged_get_stats(mem, &paged_stats) || paged_stats.admitted_pages != 0 || paged_stats.committed_pages != decoded_pages) {
        fprintf(stderr, "%s : admission release changed committed pages\n", __func__);
        llama_free(ctx);
        return 1;
    }
    if (hybrid->get_block_manager()->token_count(0) != tokens.size()) {
        fprintf(stderr, "%s : block table token count does not match decoded tokens\n", __func__);
        llama_free(ctx);
        return 1;
    }
    const llama_pos pos_before = llama_memory_seq_pos_max(mem, 0);
    if (llama_memory_seq_rm(mem, 0, 1, -1)) {
        fprintf(stderr, "%s : rollback beyond the checkpoint window unexpectedly succeeded\n", __func__);
        llama_free(ctx);
        return 1;
    }
    const llama_pos pos_after = llama_memory_seq_pos_max(mem, 0);
    if (pos_after != pos_before) {
        fprintf(stderr, "%s : failed rollback changed sequence position from %d to %d\n", __func__, pos_before, pos_after);
        llama_free(ctx);
        return 1;
    }
    if (hybrid->get_block_manager()->token_count(0) != tokens.size()) {
        fprintf(stderr, "%s : failed rollback changed block table token count\n", __func__);
        llama_free(ctx);
        return 1;
    }

    auto dense_params = cparams;
    dense_params.kv_paged_storage = false;
    llama_context * dense_ctx = llama_init_from_model(model, dense_params);
    if (dense_ctx == nullptr) {
        fprintf(stderr, "%s : failed to create dense reference context\n", __func__);
        llama_free(ctx);
        return 1;
    }
    llama_batch dense_batch = llama_batch_init(tokens.size(), 0, 1);
    for (uint32_t pos = 0; pos < tokens.size(); ++pos) {
        common_batch_add(dense_batch, tokens[pos], pos, { 0 }, pos + 1 == tokens.size());
    }
    const bool dense_decode_ok = llama_decode(dense_ctx, dense_batch) == 0;
    llama_batch_free(dense_batch);
    if (!dense_decode_ok) {
        fprintf(stderr, "%s : dense reference decode failed\n", __func__);
        llama_free(dense_ctx);
        llama_free(ctx);
        return 1;
    }

    const float * paged_logits = llama_get_logits_ith(ctx, -1);
    const float * dense_logits = llama_get_logits_ith(dense_ctx, -1);
    if (paged_logits == nullptr || dense_logits == nullptr) {
        fprintf(stderr, "%s : missing dense or paged logits\n", __func__);
        llama_free(dense_ctx);
        llama_free(ctx);
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    int paged_argmax = 0;
    int dense_argmax = 0;
    float max_abs_diff = 0.0f;
    for (int token = 0; token < n_vocab; ++token) {
        if (paged_logits[token] > paged_logits[paged_argmax]) {
            paged_argmax = token;
        }
        if (dense_logits[token] > dense_logits[dense_argmax]) {
            dense_argmax = token;
        }
        max_abs_diff = std::max(max_abs_diff, std::fabs(paged_logits[token] - dense_logits[token]));
    }
    if (paged_argmax != dense_argmax) {
        fprintf(stderr, "%s : dense/paged greedy token mismatch: %d != %d, max abs diff %g\n", __func__, paged_argmax, dense_argmax, max_abs_diff);
        llama_free(dense_ctx);
        llama_free(ctx);
        return 1;
    }
    if (max_abs_diff > 0.25f) {
        fprintf(stderr, "%s : dense/paged max abs diff %g exceeds 0.25\n", __func__, max_abs_diff);
        llama_free(dense_ctx);
        llama_free(ctx);
        return 1;
    }
    fprintf(stderr, "%s : dense/paged greedy token %d matches, max abs diff %g\n", __func__, paged_argmax, max_abs_diff);
    llama_free(dense_ctx);

    common_prompt_checkpoint checkpoint;
    checkpoint.update_tgt(ctx, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
    llama_context * restored_ctx = llama_init_from_model(model, cparams);
    if (restored_ctx == nullptr) {
        fprintf(stderr, "%s : failed to create paged restore context\n", __func__);
        llama_free(ctx);
        return 1;
    }
    checkpoint.load_tgt(restored_ctx, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (llama_memory_seq_pos_max(llama_get_memory(restored_ctx), 0) != pos_after) {
        fprintf(stderr, "%s : restored paged sequence position mismatch\n", __func__);
        llama_free(restored_ctx);
        llama_free(ctx);
        return 1;
    }

    std::vector<float> continued_logits;
    std::vector<float> restored_logits;
    if (!decode_one(ctx, tokens.back(), pos_after + 1, 0, n_vocab, continued_logits) ||
            !decode_one(restored_ctx, tokens.back(), pos_after + 1, 0, n_vocab, restored_logits)) {
        fprintf(stderr, "%s : paged checkpoint continuation failed\n", __func__);
        llama_free(restored_ctx);
        llama_free(ctx);
        return 1;
    }
    if (argmax(continued_logits) != argmax(restored_logits)) {
        fprintf(stderr, "%s : paged checkpoint continuation token mismatch\n", __func__);
        llama_free(restored_ctx);
        llama_free(ctx);
        return 1;
    }
    llama_free(restored_ctx);

    const llama_pos fork_pos = pos_after + 1;
    llama_memory_seq_cp(mem, 0, 1, -1, -1);
    const auto source_before_cow = hybrid->get_block_manager()->page_table(0);
    const auto fork_before_cow = hybrid->get_block_manager()->page_table(1);
    if (source_before_cow != fork_before_cow) {
        fprintf(stderr, "%s : fork did not share paged blocks\n", __func__);
        llama_free(ctx);
        return 1;
    }

    std::vector<float> fork_logits;
    std::vector<float> source_logits;
    if (!decode_one(ctx, tokens.back(), fork_pos + 1, 1, n_vocab, fork_logits)) {
        fprintf(stderr, "%s : fork append failed\n", __func__);
        llama_free(ctx);
        return 1;
    }
    const auto source_after_cow = hybrid->get_block_manager()->page_table(0);
    const auto fork_after_cow = hybrid->get_block_manager()->page_table(1);
    if (source_after_cow.back().id == fork_after_cow.back().id || hybrid->get_block_manager()->stats().cow_pages == 0) {
        fprintf(stderr, "%s : fork append did not copy the shared tail\n", __func__);
        llama_free(ctx);
        return 1;
    }
    if (!decode_one(ctx, tokens.back(), fork_pos + 1, 0, n_vocab, source_logits) || argmax(source_logits) != argmax(fork_logits)) {
        fprintf(stderr, "%s : source/fork continuation token mismatch\n", __func__);
        llama_free(ctx);
        return 1;
    }

    const size_t full_state_size = llama_state_get_size(ctx);
    std::vector<uint8_t> full_state(full_state_size);
    if (llama_state_get_data(ctx, full_state.data(), full_state.size()) != full_state_size) {
        fprintf(stderr, "%s : full paged state save failed\n", __func__);
        llama_free(ctx);
        return 1;
    }
    llama_context * full_restored_ctx = llama_init_from_model(model, cparams);
    if (full_restored_ctx == nullptr || llama_state_set_data(full_restored_ctx, full_state.data(), full_state.size()) != full_state_size) {
        fprintf(stderr, "%s : full paged state restore failed\n", __func__);
        llama_free(full_restored_ctx);
        llama_free(ctx);
        return 1;
    }
    auto * full_hybrid = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(full_restored_ctx));
    if (full_hybrid == nullptr) {
        fprintf(stderr, "%s : restored context has no hybrid memory\n", __func__);
        llama_free(full_restored_ctx);
        llama_free(ctx);
        return 1;
    }
    const auto restored_source = full_hybrid->get_block_manager()->page_table(0);
    const auto restored_fork = full_hybrid->get_block_manager()->page_table(1);
    if (restored_source.size() != source_after_cow.size() || restored_fork.size() != fork_after_cow.size() ||
            !(restored_source.front() == restored_fork.front()) || restored_source.back() == restored_fork.back()) {
        fprintf(stderr, "%s : full paged state did not preserve shared-page layout\n", __func__);
        llama_free(full_restored_ctx);
        llama_free(ctx);
        return 1;
    }
    llama_free(full_restored_ctx);

    fprintf(stderr, "%s : paged checkpoint restore and fork tail CoW passed\n", __func__);

    fprintf(stderr, "%s : failed hybrid rollback preserved sequence state at position %d\n", __func__, pos_after);
    llama_free(ctx);
    return 0;
}
