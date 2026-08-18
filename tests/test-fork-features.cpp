#include "ggml.h"

#include "common.h"
#include "speculative.h"

#include <cstdlib>
#include <cstring>
#include <string>

static void require(bool condition) {
    if (!condition) {
        std::abort();
    }
}

static void test_turbo_types() {
    require(std::strcmp(ggml_type_name(GGML_TYPE_TURBO2_0),   "turbo2") == 0);
    require(std::strcmp(ggml_type_name(GGML_TYPE_TURBO3_0),   "turbo3") == 0);
    require(std::strcmp(ggml_type_name(GGML_TYPE_TURBO4_0),   "turbo4") == 0);
    require(std::strcmp(ggml_type_name(GGML_TYPE_TURBO2_TCQ), "turbo2_tcq") == 0);
    require(std::strcmp(ggml_type_name(GGML_TYPE_TURBO3_TCQ), "turbo3_tcq") == 0);
}

static void test_speculative_backends() {
    require(common_speculative_type_from_name("draft-mtp") == COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
    require(common_speculative_type_from_name("draft-dflash") == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH);
    require(common_speculative_type_to_str(COMMON_SPECULATIVE_TYPE_DRAFT_MTP) == "draft-mtp");
    require(common_speculative_type_to_str(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) == "draft-dflash");

    const std::string names = common_speculative_type_name_str({
        COMMON_SPECULATIVE_TYPE_DRAFT_MTP,
        COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH,
    });
    require(names.find("draft-mtp") != std::string::npos);
    require(names.find("draft-dflash") != std::string::npos);
}

int main() {
    test_turbo_types();
    test_speculative_backends();
    return 0;
}
