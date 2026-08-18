#include "server-prefix-cache-index.h"

#include <cstdlib>

static void require(bool condition) {
    if (!condition) {
        std::abort();
    }
}

static std::vector<llama_token> make_tokens(size_t count, llama_token first = 0) {
    std::vector<llama_token> result(count);
    for (size_t i = 0; i < count; ++i) {
        result[i] = first + static_cast<llama_token>(i);
    }
    return result;
}

int main() {
    server_prefix_cache_index index(16);
    const auto short_prefix = make_tokens(17, 100);
    const auto long_prefix = make_tokens(33, 100);
    auto query = make_tokens(40, 100);

    require(index.insert(1, short_prefix, "model-a"));
    require(index.insert(2, long_prefix, "model-a"));
    require(index.insert(3, short_prefix, "model-b"));
    require(!index.insert(2, short_prefix, "model-a"));

    size_t matched = 0;
    require(index.find_longest_prefix(query, "model-a", &matched) == 2);
    require(matched == 33);
    require(index.find_longest_prefix(query, "model-b", &matched) == 3);
    require(matched == 17);

    query[16] = -1;
    require(index.find_longest_prefix(query, "model-a", &matched) == 0);
    require(matched == 0);

    require(index.erase(2));
    query = make_tokens(40, 100);
    require(index.find_longest_prefix(query, "model-a", &matched) == 1);
    require(matched == 17);
    require(!index.erase(2));
    require(index.size() == 2);
    return 0;
}
