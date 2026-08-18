#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

enum class server_schedule_phase {
    decode,
    prefill,
};

struct server_schedule_request {
    int id = -1;
    int task_id = -1;
    server_schedule_phase phase = server_schedule_phase::prefill;
    uint64_t ready_since_us = 0;
    uint64_t last_progress_us = 0;
    size_t remaining_tokens = 0;
    bool interactive = false;
};

struct server_schedule_allocation {
    int id = -1;
    size_t tokens = 0;
};

struct server_schedule_plan {
    std::vector<int> decode_order;
    std::vector<server_schedule_allocation> prefill;
    uint64_t deadline_misses = 0;
    uint64_t max_prefill_wait_us = 0;
};

class server_continuous_scheduler {
public:
    struct config {
        size_t max_batched_tokens = 2048;
        size_t block_size = 16;
        size_t max_prefill_chunk = 512;
        size_t minimum_prefill_progress = 16;
        uint64_t target_step_us = 10000;
        uint64_t decode_deadline_us = 80000;
        uint64_t max_prefill_wait_us = 100000;
        size_t pipeline_prefill_floor = 0;
    };

    explicit server_continuous_scheduler(config params);

    server_schedule_plan plan(const std::vector<server_schedule_request> & requests, uint64_t now_us);
    void commit_prefill(int id, int task_id, size_t tokens, uint64_t now_us);
    void observe(size_t prefill_tokens, size_t decode_tokens, uint64_t elapsed_us);

    double prefill_us_per_token() const;
    double decode_us_per_token() const;

private:
    struct request_state {
        int task_id = -1;
        double virtual_runtime = 0.0;
        double weight = 1.0;
        uint64_t last_prefill_us = 0;
    };

    config params;
    std::unordered_map<int, request_state> states;
    double prefill_cost_us = 1000.0;
    double decode_cost_us = 10000.0;
};
