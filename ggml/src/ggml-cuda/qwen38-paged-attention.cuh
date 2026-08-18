#pragma once

#include "common.cuh"

void ggml_cuda_op_qwen38_paged_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
