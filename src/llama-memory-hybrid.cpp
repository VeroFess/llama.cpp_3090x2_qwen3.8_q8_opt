#include "llama-memory-hybrid.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-context.h"

#include <limits>
#include <stdexcept>
#include <unordered_map>

static uint32_t llama_paged_page_count(uint32_t kv_size, uint32_t page_size) {
    if (page_size == 0 || kv_size % page_size != 0) {
        throw std::runtime_error("invalid hybrid paged KV page size");
    }
    return kv_size / page_size;
}

static int32_t llama_paged_block_table_bucket(int32_t blocks) {
    if (blocks <= 256) {
        return 256;
    }
    if (blocks <= 2048) {
        return 2048;
    }
    if (blocks <= 8192) {
        return 8192;
    }
    if (blocks <= 16384) {
        return 16384;
    }
    return blocks;
}

//
// llama_memory_hybrid
//

llama_memory_hybrid::llama_memory_hybrid(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   page_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                     bool   paged_storage,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr) :
    hparams(model.hparams),
    mem_attn(paged_storage ? nullptr : new llama_kv_cache(
        model,
        model.hparams,
        type_k,
        type_v,
        v_trans,
        offload,
        unified,
        kv_size,
        n_seq_max,
        n_pad,
        n_swa,
        swa_type,
        nullptr,
        filter_attn == nullptr ?
            [&](int32_t il) { return !hparams.is_recr(il); }
            : filter_attn,
        nullptr,
        nullptr
    )),
    mem_recr(new llama_memory_recurrent(
        model,
        type_r,
        type_s,
        offload,
        rs_size,
        n_seq_max,
        n_rs_seq,
        filter_recr == nullptr ?
            [&](int32_t il) { return hparams.is_recr(il); }
            : filter_recr
    )),
    block_manager(page_size == 0 ? nullptr : new llama_paged_block_manager()),
    paged_pool(paged_storage ? new llama_paged_kv_pool(model, type_k, page_size, llama_paged_page_count(kv_size, page_size), offload,
        filter_attn == nullptr ? llama_memory_i::layer_filter_cb([&](int32_t il) { return !hparams.is_recr(il); }) : filter_attn) : nullptr) {
    if (paged_storage && (page_size == 0 || type_k != type_v || type_k != GGML_TYPE_Q8_0 || kv_size % page_size != 0)) {
        throw std::runtime_error("invalid hybrid paged KV configuration");
    }
    if (block_manager) {
        block_manager->configure(kv_size, page_size, n_seq_max);
    }
}

llama_memory_context_ptr llama_memory_hybrid::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            // DFlash target models need per-seq ubatches so the per-ubatch slot
            // switch in llama_context::decode() can route hidden-state capture
            // and tape writes to the correct slot.
            if (embd_all || force_split_seq) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = paged_pool != nullptr || mem_attn->get_n_stream() == 1;

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = mem_recr->n_rs_seq;

                ubatch = balloc.split_equal(
                    n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0,
                    paged_pool != nullptr ? (paged_pool->tensor_sharded() ? 8 : 4) : std::numeric_limits<uint32_t>::max());
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!mem_recr->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        llama_kv_cache::slot_info_vec_t heads_attn;
        if (mem_attn) {
            heads_attn = mem_attn->prepare(ubatches);
            if (heads_attn.empty()) {
                LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
                return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
            }
        }

        std::vector<std::vector<std::pair<int, size_t>>> block_requests(ubatches.size());
        if (block_manager) {
            std::unordered_map<int, size_t> running_tokens;
            for (size_t i_ubatch = 0; i_ubatch < ubatches.size(); ++i_ubatch) {
                const auto & ubatch = ubatches[i_ubatch];
                std::unordered_map<int, size_t> requests;
                for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                    if (ubatch.pos[i] < 0) {
                        continue;
                    }
                    const size_t tokens = static_cast<size_t>(ubatch.pos[i]) + 1;
                    for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
                        const int sequence_id = ubatch.seq_id[i][j];
                        auto running = running_tokens.find(sequence_id);
                        if (running == running_tokens.end()) {
                            running = running_tokens.emplace(sequence_id, block_manager->token_count(sequence_id)).first;
                        }
                        running->second = std::max(running->second, tokens);
                        requests[sequence_id] = running->second;
                    }
                }
                block_requests[i_ubatch].assign(requests.begin(), requests.end());
            }

            std::vector<std::pair<int, size_t>> final_requests;
            final_requests.reserve(running_tokens.size());
            for (const auto & request : running_tokens) {
                final_requests.push_back(request);
            }
            if (!block_manager->can_reserve_batch(final_requests)) {
                LLAMA_LOG_ERROR("%s: paged block reservation failed\n", __func__);
                return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
            }
        }

        return std::make_unique<llama_memory_hybrid_context>(
                this, std::move(heads_attn), std::move(ubatches), std::move(block_requests));
    } while(false);

    return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid::init_full() {
    return std::make_unique<llama_memory_hybrid_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_context>(this, lctx, optimize);
}

bool llama_memory_hybrid::get_can_shift() const {
    return mem_attn != nullptr && mem_attn->get_can_shift();
}

void llama_memory_hybrid::clear(bool data) {
    if (mem_attn) {
        mem_attn->clear(data);
    } else if (data && paged_pool) {
        paged_pool->clear();
    }
    mem_recr->clear(data);
    if (block_manager) {
        block_manager->clear();
    }
}

bool llama_memory_hybrid::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (!mem_recr->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    if (mem_attn && !mem_attn->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    if (block_manager) {
        if (seq_id < 0) {
            const llama_pos range_begin = p0 < 0 ? 0 : p0;
            const llama_pos range_end = p1 < 0 ? std::numeric_limits<llama_pos>::max() : p1;
            if (range_begin != range_end) {
                block_manager->clear();
            }
        } else {
            const llama_pos pos_max = seq_pos_max(seq_id);
            if (!block_manager->reserve(seq_id, pos_max < 0 ? 0 : static_cast<size_t>(pos_max) + 1)) {
                LLAMA_LOG_ERROR("%s: failed to update paged block table for sequence %d\n", __func__, seq_id);
                return false;
            }
        }
    }
    return true;
}

void llama_memory_hybrid::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (mem_attn) {
        mem_attn->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }
    if (paged_pool) {
        const llama_pos range_begin = p0 < 0 ? 0 : p0;
        const llama_pos range_end = p1 < 0 ? std::numeric_limits<llama_pos>::max() : p1;
        if (range_begin != 0 || range_end != std::numeric_limits<llama_pos>::max() || block_manager->token_count(seq_id_dst) != 0 || !block_manager->fork_sequence(seq_id_src, seq_id_dst)) {
            throw std::runtime_error("paged KV only supports copying a full sequence into an empty destination");
        }
    }
    mem_recr->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_hybrid::seq_keep(llama_seq_id seq_id) {
    if (mem_attn) {
        mem_attn->seq_keep(seq_id);
    } else if (block_manager) {
        block_manager->keep(seq_id);
    }
    mem_recr->seq_keep(seq_id);
}

void llama_memory_hybrid::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (mem_attn) {
        mem_attn->seq_add(seq_id, p0, p1, shift);
    } else {
        throw std::runtime_error("paged KV does not support position shifts");
    }
    mem_recr->seq_add(seq_id, p0, p1, shift);
}

void llama_memory_hybrid::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (mem_attn) {
        mem_attn->seq_div(seq_id, p0, p1, d);
    } else {
        throw std::runtime_error("paged KV does not support position division");
    }
    mem_recr->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_hybrid::seq_pos_min(llama_seq_id seq_id) const {
    const llama_pos recurrent = mem_recr->seq_pos_min(seq_id);
    if (!mem_attn && block_manager) {
        return block_manager->token_count(seq_id) == 0 ? -1 : recurrent;
    }
    return std::max(mem_attn->seq_pos_min(seq_id), recurrent);
}

llama_pos llama_memory_hybrid::seq_pos_max(llama_seq_id seq_id) const {
    const llama_pos recurrent = mem_recr->seq_pos_max(seq_id);
    if (!mem_attn && block_manager) {
        const size_t tokens = block_manager->token_count(seq_id);
        return tokens == 0 ? -1 : std::min<llama_pos>(static_cast<llama_pos>(tokens - 1), recurrent);
    }
    return std::min(mem_attn->seq_pos_max(seq_id), recurrent);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = mem_attn ? mem_attn->memory_breakdown() : paged_pool->memory_breakdown();
    for (const auto & buft_size : mem_recr->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

void llama_memory_hybrid::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_attn) {
            mem_attn->state_write(io, seq_id, flags);
        } else {
            paged_pool->state_write(io, *block_manager, seq_id);
        }
    }
    mem_recr->state_write(io, seq_id, flags);
}

void llama_memory_hybrid::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_attn) {
                mem_attn->state_read(io, seq_id, flags);
            } else {
                paged_pool->state_read(io, *block_manager, seq_id);
            }
        }
        mem_recr->state_read(io, seq_id, flags);
    } catch (...) {
        if (!mem_attn && block_manager) {
            if (seq_id < 0) {
                block_manager->clear();
            } else {
                block_manager->reserve(seq_id, 0);
            }
        }
        mem_recr->seq_rm(seq_id, -1, -1);
        throw;
    }
}

llama_kv_cache * llama_memory_hybrid::get_mem_attn() const {
    return mem_attn.get();
}

llama_memory_recurrent * llama_memory_hybrid::get_mem_recr() const {
    return mem_recr.get();
}

llama_paged_block_manager * llama_memory_hybrid::get_block_manager() const {
    return block_manager.get();
}

llama_paged_kv_pool * llama_memory_hybrid::get_paged_pool() const {
    return paged_pool.get();
}

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_status status) : status(status) {}

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_hybrid * mem) :
    ctx_attn(mem->get_mem_attn() ? mem->get_mem_attn()->init_full() : nullptr),
    ctx_recr(mem->get_mem_recr()->init_full()),
    block_manager(mem->get_block_manager()),
    paged_pool(mem->get_paged_pool()),
    status(ctx_attn ? llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status()) : ctx_recr->get_status()) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
        llama_memory_hybrid * mem,
              llama_context * lctx,
                       bool   optimize) :
    ctx_attn(mem->get_mem_attn() ? mem->get_mem_attn()->init_update(lctx, optimize) : nullptr),
    ctx_recr(mem->get_mem_recr()->init_update(lctx, optimize)),
    block_manager(mem->get_block_manager()),
    paged_pool(mem->get_paged_pool()),
    status(ctx_attn ? llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status()) : ctx_recr->get_status()) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
              llama_memory_hybrid * mem,
                  slot_info_vec_t   sinfos_attn,
        std::vector<llama_ubatch>   ubatches,
        std::vector<std::vector<std::pair<int, size_t>>> block_requests) :
    ubatches(std::move(ubatches)),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_attn(mem->get_mem_attn() ? new llama_kv_cache_context(mem->get_mem_attn(), std::move(sinfos_attn), this->ubatches) : nullptr),
    ctx_recr(new llama_memory_recurrent_context(mem->get_mem_recr(), this->ubatches)),
    block_manager(mem->get_block_manager()),
    paged_pool(mem->get_paged_pool()),
    block_requests(std::move(block_requests)),
    paged_metadata(this->ubatches.size()),
    status(ctx_attn ? llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status()) : ctx_recr->get_status()) {
}

bool llama_memory_hybrid_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (ctx_attn) {
        ctx_attn->next();
    }
    ctx_recr->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_hybrid_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    if (ctx_attn) {
        res = res & ctx_attn->apply();
    }
    res = res & ctx_recr->apply();

    if (res && block_manager && !block_requests[i_next].empty()) {
        res = block_manager->reserve_batch(block_requests[i_next]);
    }
    if (res && paged_pool && block_manager) {
        paged_pool->copy_pages(block_manager->take_pending_copies());
    }
    if (res && paged_pool) {
        res = build_paged_metadata();
    }

    return res;
}

llama_memory_status llama_memory_hybrid_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_hybrid_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const llama_kv_cache_context * llama_memory_hybrid_context::get_attn() const {
    return static_cast<const llama_kv_cache_context *>(ctx_attn.get());
}

const llama_paged_kv_pool * llama_memory_hybrid_context::get_paged_pool() const {
    return paged_pool;
}

const llama_paged_kv_metadata * llama_memory_hybrid_context::get_paged_metadata() const {
    return paged_pool && i_next < paged_metadata.size() ? &paged_metadata[i_next] : nullptr;
}

bool llama_memory_hybrid_context::build_paged_metadata() {
    if (!paged_pool || !block_manager || i_next >= ubatches.size()) {
        return false;
    }

    const llama_ubatch & ubatch = ubatches[i_next];
    auto & metadata = paged_metadata[i_next];
    metadata = {};
    metadata.n_sequences = static_cast<int32_t>(ubatch.n_seqs_unq);
    if (metadata.n_sequences <= 0 || paged_pool->n_devices() != 2) {
        return false;
    }

    std::unordered_map<int, int32_t> rows;
    std::vector<std::vector<llama_paged_block_handle>> tables(metadata.n_sequences);
    for (int32_t row = 0; row < metadata.n_sequences; ++row) {
        const int sequence_id = ubatch.seq_id_unq[row];
        rows[sequence_id] = row;
        tables[row] = block_manager->page_table(sequence_id);
        metadata.max_blocks = std::max(metadata.max_blocks, static_cast<int32_t>(tables[row].size()));
        metadata.context_lens.push_back(static_cast<int32_t>(block_manager->token_count(sequence_id)));
    }
    if (metadata.max_blocks <= 0) {
        return false;
    }
    metadata.max_blocks = llama_paged_block_table_bucket(metadata.max_blocks);

    metadata.batch_offsets.assign(metadata.n_sequences, -1);
    metadata.batch_lens.assign(metadata.n_sequences, 0);
    for (auto & values : metadata.block_tables) {
        values.assign(static_cast<size_t>(metadata.n_sequences) * metadata.max_blocks, -1);
    }
    for (int32_t row = 0; row < metadata.n_sequences; ++row) {
        for (size_t page = 0; page < tables[row].size(); ++page) {
            for (uint32_t device = 0; device < 2; ++device) {
                metadata.block_tables[device][static_cast<size_t>(row) * metadata.max_blocks + page] = tables[row][page].physical[device];
            }
        }
    }

    for (auto & values : metadata.write_slots) {
        values.resize(ubatch.n_tokens);
    }
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.n_seq_id[i] != 1 || ubatch.pos[i] < 0) {
            return false;
        }
        const int sequence_id = ubatch.seq_id[i][0];
        const auto row_it = rows.find(sequence_id);
        if (row_it == rows.end()) {
            return false;
        }
        const int32_t row = row_it->second;
        if (metadata.batch_offsets[row] < 0) {
            metadata.batch_offsets[row] = i;
        } else if (metadata.batch_offsets[row] + metadata.batch_lens[row] != static_cast<int32_t>(i)) {
            return false;
        }
        ++metadata.batch_lens[row];

        const size_t logical_page = static_cast<size_t>(ubatch.pos[i]) / paged_pool->page_size();
        if (logical_page >= tables[row].size()) {
            return false;
        }
        const uint32_t offset = static_cast<uint32_t>(ubatch.pos[i]) % paged_pool->page_size();
        for (uint32_t device = 0; device < 2; ++device) {
            metadata.write_slots[device][i] = tables[row][logical_page].physical[device] * paged_pool->page_size() + offset;
        }
    }
    return true;
}

ggml_tensor * llama_memory_hybrid_context::get_turbo_rot_forward() const {
    return ctx_attn ? ctx_attn->get_turbo_rot_forward() : nullptr;
}

ggml_tensor * llama_memory_hybrid_context::get_turbo_rot_inverse() const {
    return ctx_attn ? ctx_attn->get_turbo_rot_inverse() : nullptr;
}

const llama_memory_recurrent_context * llama_memory_hybrid_context::get_recr() const {
    return static_cast<const llama_memory_recurrent_context *>(ctx_recr.get());
}
