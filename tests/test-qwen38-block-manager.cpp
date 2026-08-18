#include "src/llama-paged-block-manager.h"

#include <cstdlib>

static void require(bool condition) {
    if (!condition) {
        std::abort();
    }
}

static void test_dual_gpu_atomic_allocation() {
    llama_paged_block_manager manager;
    manager.configure_physical({ 4, 3 }, 16, 4);

    require(manager.reserve(0, 32));
    const auto before = manager.stats();
    require(before.free_logical_pages == 1);
    require(before.free_physical_pages[0] == 2);
    require(before.free_physical_pages[1] == 1);

    require(!manager.reserve(1, 32));
    const auto failed = manager.stats();
    require(failed.free_logical_pages == before.free_logical_pages);
    require(failed.free_physical_pages == before.free_physical_pages);
    require(failed.resident_sequences == before.resident_sequences);

    require(manager.reserve(1, 16));
    manager.release(0);
    manager.release(1);
    const auto released = manager.stats();
    require(released.free_logical_pages == released.total_logical_pages);
    require(released.free_physical_pages == released.total_physical_pages);
}

static void test_generation_reuse() {
    llama_paged_block_manager manager;
    manager.configure_physical({ 1, 1 }, 16, 2);
    require(manager.reserve(0, 16));
    const auto first = manager.page_table(0).front();
    manager.release(0);
    require(manager.reserve(1, 16));
    const auto second = manager.page_table(1).front();
    require(second.id == first.id);
    require(second.physical == first.physical);
    require(second.generation != first.generation);
}

static void test_fork_tail_cow() {
    llama_paged_block_manager manager;
    manager.configure_physical({ 4, 4 }, 16, 4);
    require(manager.reserve(0, 32));
    require(manager.fork_sequence(0, 1));

    const auto source = manager.page_table(0);
    const auto forked = manager.page_table(1);
    require(source == forked);
    require(manager.ensure_writable_tail(1));

    const auto copied = manager.page_table(1);
    require(copied[0].id == source[0].id);
    require(copied[1].id != source[1].id);
    require(manager.stats().cow_pages == 1);
    const auto copies = manager.take_pending_copies();
    require(copies.size() == 1);
    require(copies[0].source == source[1]);
    require(copies[0].destination == copied[1]);

    manager.release(0);
    manager.release(1);
    require(manager.stats().free_logical_pages == manager.stats().total_logical_pages);
}

static void test_append_cow_only_for_partial_tail() {
    llama_paged_block_manager manager;
    manager.configure_physical({ 6, 6 }, 16, 4);
    require(manager.reserve(0, 17));
    require(manager.token_count(0) == 17);
    require(manager.fork_sequence(0, 1));
    require(manager.reserve(1, 18));
    require(manager.stats().cow_pages == 1);
    require(manager.page_table(0).back().id != manager.page_table(1).back().id);

    manager.release(0);
    manager.release(1);
    manager.configure_physical({ 6, 6 }, 16, 4);
    require(manager.reserve(0, 16));
    require(manager.fork_sequence(0, 1));
    require(manager.reserve(1, 17));
    require(manager.stats().cow_pages == 0);
    require(manager.page_table(0).front().id == manager.page_table(1).front().id);
}

static void test_batch_reservation_is_atomic() {
    llama_paged_block_manager manager;
    manager.configure_physical({ 3, 3 }, 16, 4);

    const std::vector<std::pair<int, size_t>> too_large = { { 0, 32 }, { 1, 32 } };
    require(!manager.can_reserve_batch(too_large));
    require(!manager.reserve_batch(too_large));
    require(manager.stats().resident_sequences == 0);
    require(manager.stats().free_logical_pages == 3);

    const std::vector<std::pair<int, size_t>> fits = { { 0, 16 }, { 1, 32 } };
    require(manager.can_reserve_batch(fits));
    require(manager.reserve_batch(fits));
    require(manager.stats().resident_sequences == 2);
    require(manager.stats().free_logical_pages == 0);
}

static void test_admission_quota_does_not_allocate_pages() {
    llama_paged_block_manager manager;
    manager.configure_physical({ 4, 4 }, 16, 4);

    require(manager.admit(0, 48));
    auto stats = manager.stats();
    require(stats.admitted_pages == 3);
    require(stats.committed_pages == 3);
    require(stats.free_logical_pages == 4);

    require(manager.reserve(0, 16));
    stats = manager.stats();
    require(stats.committed_pages == 3);
    require(stats.free_logical_pages == 3);
    require(!manager.admit(1, 32));

    manager.release_admission(0);
    stats = manager.stats();
    require(stats.admitted_pages == 0);
    require(stats.committed_pages == 1);
    require(manager.admit(1, 32));
    require(manager.stats().committed_pages == 3);
}

static void test_prefix_pin_attach_and_evict() {
    llama_paged_block_manager manager;
    manager.configure_physical({ 6, 6 }, 16, 4);
    require(manager.reserve(0, 48));

    llama_paged_prefix_handle prefix;
    require(manager.pin_prefix(0, 32, prefix));
    require(prefix.id != 0);
    require(manager.prefix_token_count(prefix) == 32);
    require(manager.stats().cached_prefixes == 1);
    require(manager.stats().cached_prefix_pages == 2);

    const auto source = manager.page_table(0);
    manager.release(0);
    require(manager.stats().free_logical_pages == 4);
    require(manager.attach_prefix(1, prefix));
    const auto attached = manager.page_table(1);
    require(attached.size() == 2);
    require(attached[0] == source[0]);
    require(attached[1] == source[1]);
    require(manager.token_count(1) == 32);

    require(manager.reserve(1, 33));
    require(manager.page_table(1).size() == 3);
    require(manager.stats().cow_pages == 0);
    require(manager.release_prefix(prefix));
    require(manager.prefix_token_count(prefix) == 0);
    require(!manager.attach_prefix(2, prefix));
    manager.release(1);
    require(manager.stats().free_logical_pages == manager.stats().total_logical_pages);
}

static void test_prefix_partial_tail_uses_cow() {
    llama_paged_block_manager manager;
    manager.configure_physical({ 4, 4 }, 16, 4);
    require(manager.reserve(0, 17));

    llama_paged_prefix_handle prefix;
    require(manager.pin_prefix(0, 17, prefix));
    const auto source = manager.page_table(0);
    manager.release(0);
    require(manager.attach_prefix(1, prefix));
    require(manager.reserve(1, 18));
    require(manager.page_table(1).back().id != source.back().id);
    require(manager.stats().cow_pages == 1);
    require(manager.take_pending_copies().size() == 1);
    require(manager.release_prefix(prefix));
    manager.release(1);

    prefix = {};
    require(!manager.pin_prefix(0, 32, prefix));
    require(manager.stats().free_logical_pages == manager.stats().total_logical_pages);
}

int main() {
    test_dual_gpu_atomic_allocation();
    test_generation_reuse();
    test_fork_tail_cow();
    test_append_cow_only_for_partial_tail();
    test_batch_reservation_is_atomic();
    test_admission_quota_does_not_allocate_pages();
    test_prefix_pin_attach_and_evict();
    test_prefix_partial_tail_uses_cow();
    return 0;
}
