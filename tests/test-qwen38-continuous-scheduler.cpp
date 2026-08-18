#include "server-continuous-scheduler.h"

#include <cstdlib>

static void require(bool condition) {
    if (!condition) {
        std::abort();
    }
}

static server_continuous_scheduler make_scheduler() {
    return server_continuous_scheduler({ 64, 16, 48, 16, 50000, 80000, 100000 });
}

int main() {
    {
        auto scheduler = make_scheduler();
        const std::vector<server_schedule_request> requests = {
            { 1, 1, server_schedule_phase::decode, 10, 90000, 0, false },
            { 2, 2, server_schedule_phase::decode, 10, 20000, 0, false },
        };
        const auto plan = scheduler.plan(requests, 200000);
        require(plan.decode_order == std::vector<int>({ 2, 1 }));
        require(plan.deadline_misses == 2);
    }

    {
        auto scheduler = make_scheduler();
        const std::vector<server_schedule_request> requests = {
            { 0, 10, server_schedule_phase::decode, 1, 199000, 0, false },
            { 1, 11, server_schedule_phase::prefill, 1, 1, 128, false },
            { 2, 12, server_schedule_phase::prefill, 1, 1, 128, false },
            { 3, 13, server_schedule_phase::prefill, 1, 1, 128, false },
        };
        const auto plan = scheduler.plan(requests, 200000);
        require(plan.prefill.size() == 3);
        require(plan.prefill[0].tokens == 16);
        require(plan.prefill[1].tokens == 16);
        require(plan.prefill[2].tokens == 16);
        require(plan.max_prefill_wait_us >= 100000);
    }

    {
        server_continuous_scheduler scheduler({ 17, 16, 16, 16, 10000, 80000, 100000 });
        std::vector<server_schedule_request> requests = {
            { 0, 20, server_schedule_phase::decode, 1000, 1000, 0, false },
            { 1, 21, server_schedule_phase::prefill, 1000, 1000, 128, false },
            { 2, 22, server_schedule_phase::prefill, 1000, 1000, 128, true },
        };
        auto plan = scheduler.plan(requests, 2000);
        require(plan.prefill.front().id == 2);
        scheduler.commit_prefill(2, 22, plan.prefill.front().tokens, 2000);
        plan = scheduler.plan(requests, 3000);
        require(plan.prefill.front().id == 1);
    }

    {
        auto scheduler = make_scheduler();
        scheduler.observe(100, 0, 200000);
        require(scheduler.prefill_us_per_token() > 1000.0);
        scheduler.observe(0, 2, 40000);
        require(scheduler.decode_us_per_token() > 10000.0);
    }

    {
        server_continuous_scheduler scheduler({ 256, 16, 256, 16, 10000, 80000, 100000, 128 });
        std::vector<server_schedule_request> requests = {
            { 0, 30, server_schedule_phase::prefill, 1000, 1000, 256, false },
            { 1, 31, server_schedule_phase::prefill, 1000, 1000, 256, false },
        };
        auto plan = scheduler.plan(requests, 2000);
        size_t tokens = 0;
        for (const auto & allocation : plan.prefill) {
            tokens += allocation.tokens;
        }
        require(tokens == 128);

        requests.push_back({ 2, 32, server_schedule_phase::decode, 1000, 1000, 0, false });
        plan = scheduler.plan(requests, 3000);
        tokens = 0;
        for (const auto & allocation : plan.prefill) {
            tokens += allocation.tokens;
        }
        require(tokens == 16);
    }
    return 0;
}
