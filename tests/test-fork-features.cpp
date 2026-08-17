#include "ggml.h"

#include "common.h"
#include "speculative.h"

#include <cassert>
#include <cstring>
#include <string>

static void test_turbo_types() {
    assert(std::strcmp(ggml_type_name(GGML_TYPE_TURBO2_0),   "turbo2") == 0);
    assert(std::strcmp(ggml_type_name(GGML_TYPE_TURBO3_0),   "turbo3") == 0);
    assert(std::strcmp(ggml_type_name(GGML_TYPE_TURBO4_0),   "turbo4") == 0);
    assert(std::strcmp(ggml_type_name(GGML_TYPE_TURBO2_TCQ), "turbo2_tcq") == 0);
    assert(std::strcmp(ggml_type_name(GGML_TYPE_TURBO3_TCQ), "turbo3_tcq") == 0);
}

static void test_speculative_backends() {
    assert(common_speculative_type_from_name("mtp") == COMMON_SPECULATIVE_TYPE_MTP);
    assert(common_speculative_type_from_name("dflash") == COMMON_SPECULATIVE_TYPE_DFLASH);
    assert(common_speculative_type_to_str(COMMON_SPECULATIVE_TYPE_MTP) == "mtp");
    assert(common_speculative_type_to_str(COMMON_SPECULATIVE_TYPE_DFLASH) == "dflash");

    const std::string names = common_speculative_type_name_str();
    assert(names.find("mtp") != std::string::npos);
    assert(names.find("dflash") != std::string::npos);
}

int main() {
    test_turbo_types();
    test_speculative_backends();
    return 0;
}
