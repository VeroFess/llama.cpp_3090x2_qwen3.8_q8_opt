#include "server-continuous-scheduler.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

server_continuous_scheduler::server_continuous_scheduler(config params) : params(params) {
    this->params.max_batched_tokens = std::max<size_t>(1, this->params.max_batched_tokens);
    this->params.block_size = std::max<size_t>(1, this->params.block_size);
    this->params.max_prefill_chunk = std::max(this->params.block_size, this->params.max_prefill_chunk);
    this->params.minimum_prefill_progress = std::max(this->params.block_size, this->params.minimum_prefill_progress);
    this->params.target_step_us = std::max<uint64_t>(1, this->params.target_step_us);
    this->params.pipeline_prefill_floor = this->params.pipeline_prefill_floor / this->params.block_size * this->params.block_size;
}

server_schedule_plan server_continuous_scheduler::plan(const std::vector<server_schedule_request> & requests, uint64_t now_us) {
    server_schedule_plan result;
    std::unordered_set<int> active;
    std::vector<const server_schedule_request *> prefills;

    for (const auto & request : requests) {
        active.insert(request.id);
        auto & state = states[request.id];
        if (state.task_id != request.task_id) {
            state = {};
            state.task_id = request.task_id;
            state.last_prefill_us = request.ready_since_us;
        }

        if (request.phase == server_schedule_phase::decode) {
            result.decode_order.push_back(request.id);
            const uint64_t deadline = request.last_progress_us + params.decode_deadline_us;
            if (request.last_progress_us != 0 && now_us > deadline) {
                ++result.deadline_misses;
            }
        } else if (request.remaining_tokens > 0) {
            state.weight = request.interactive ? 2.0 : 1.0;
            prefills.push_back(&request);
            const uint64_t since = std::max(request.ready_since_us, state.last_prefill_us);
            if (now_us > since) {
                result.max_prefill_wait_us = std::max(result.max_prefill_wait_us, now_us - since);
            }
        }
    }

    for (auto it = states.begin(); it != states.end();) {
        if (active.find(it->first) == active.end()) {
            it = states.erase(it);
        } else {
            ++it;
        }
    }

    std::sort(result.decode_order.begin(), result.decode_order.end(), [&](int a, int b) {
        const auto find_request = [&](int id) {
            return std::find_if(requests.begin(), requests.end(), [&](const server_schedule_request & request) { return request.id == id; });
        };
        const auto ra = find_request(a);
        const auto rb = find_request(b);
        const uint64_t da = ra->last_progress_us + params.decode_deadline_us;
        const uint64_t db = rb->last_progress_us + params.decode_deadline_us;
        return da == db ? ra->ready_since_us < rb->ready_since_us : da < db;
    });

    std::sort(prefills.begin(), prefills.end(), [&](const server_schedule_request * a, const server_schedule_request * b) {
        const auto & sa = states[a->id];
        const auto & sb = states[b->id];
        const uint64_t wa = now_us - std::min(now_us, std::max(a->ready_since_us, sa.last_prefill_us));
        const uint64_t wb = now_us - std::min(now_us, std::max(b->ready_since_us, sb.last_prefill_us));
        const bool starved_a = wa >= params.max_prefill_wait_us;
        const bool starved_b = wb >= params.max_prefill_wait_us;
        if (starved_a != starved_b) {
            return starved_a;
        }
        if (a->interactive != b->interactive) {
            return a->interactive;
        }
        if (sa.virtual_runtime != sb.virtual_runtime) {
            return sa.virtual_runtime < sb.virtual_runtime;
        }
        if (a->ready_since_us != b->ready_since_us) {
            return a->ready_since_us < b->ready_since_us;
        }
        return a->id < b->id;
    });

    const size_t decode_tokens = result.decode_order.size();
    size_t budget = decode_tokens >= params.max_batched_tokens ? 0 : params.max_batched_tokens - decode_tokens;
    const size_t time_budget = std::max<size_t>(params.block_size, static_cast<size_t>(params.target_step_us / std::max(1.0, prefill_cost_us)));
    const size_t time_budget_aligned = std::max(params.block_size, time_budget / params.block_size * params.block_size);
    const size_t step_multiplier = result.decode_order.empty() ? 4 : 1;
    size_t prefill_cap = std::min(params.max_prefill_chunk, time_budget_aligned * step_multiplier);
    if (result.decode_order.empty() && prefills.size() >= 2) {
        prefill_cap = std::max(prefill_cap, std::min(params.max_prefill_chunk, params.pipeline_prefill_floor));
    }
    budget = std::min(budget, prefill_cap);

    std::unordered_map<int, size_t> allocated;
    std::unordered_map<int, double> virtual_runtime;
    std::vector<int> allocation_order;
    for (const auto * request : prefills) {
        virtual_runtime[request->id] = states[request->id].virtual_runtime;
    }
    while (budget > 0 && !prefills.empty()) {
        const server_schedule_request * selected = nullptr;
        for (const auto * request : prefills) {
            const size_t current = allocated[request->id];
            if (current >= request->remaining_tokens || current >= params.max_prefill_chunk) {
                continue;
            }
            if (selected == nullptr || virtual_runtime[request->id] < virtual_runtime[selected->id]) {
                selected = request;
            }
        }
        if (selected == nullptr) {
            break;
        }
        const size_t current = allocated[selected->id];
        const auto & state = states[selected->id];
        const uint64_t since = std::max(selected->ready_since_us, state.last_prefill_us);
        const bool starved = now_us - std::min(now_us, since) >= params.max_prefill_wait_us;
        const size_t request_quantum = starved ? params.minimum_prefill_progress : params.block_size;
        const size_t quantum = std::min({ request_quantum, budget, selected->remaining_tokens - current, params.max_prefill_chunk - current });
        if (allocated[selected->id] == 0) {
            allocation_order.push_back(selected->id);
        }
        allocated[selected->id] += quantum;
        virtual_runtime[selected->id] += quantum / state.weight;
        budget -= quantum;
    }

    for (int id : allocation_order) {
        const size_t tokens = allocated[id];
        if (tokens > 0) {
            result.prefill.push_back({ id, tokens });
        }
    }
    return result;
}

void server_continuous_scheduler::commit_prefill(int id, int task_id, size_t tokens, uint64_t now_us) {
    auto & state = states[id];
    if (state.task_id != task_id) {
        state = {};
        state.task_id = task_id;
    }
    state.virtual_runtime += tokens / state.weight;
    state.last_prefill_us = now_us;
}

void server_continuous_scheduler::observe(size_t prefill_tokens, size_t decode_tokens, uint64_t elapsed_us) {
    constexpr double alpha = 0.2;
    if (prefill_tokens > 0 && decode_tokens == 0) {
        const double sample = static_cast<double>(elapsed_us) / prefill_tokens;
        prefill_cost_us = (1.0 - alpha) * prefill_cost_us + alpha * sample;
    } else if (decode_tokens > 0 && prefill_tokens == 0) {
        const double sample = static_cast<double>(elapsed_us) / decode_tokens;
        decode_cost_us = (1.0 - alpha) * decode_cost_us + alpha * sample;
    }
}

double server_continuous_scheduler::prefill_us_per_token() const {
    return prefill_cost_us;
}

double server_continuous_scheduler::decode_us_per_token() const {
    return decode_cost_us;
}
