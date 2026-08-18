#pragma once

#include "common.h"

#include <string>

struct llama_context;
struct llama_model;

bool server_qwen38_profile_enabled(const common_params & params);
bool server_qwen38_profile_apply(common_params & params, std::string & error);
bool server_qwen38_profile_validate_hardware(common_params & params, std::string & error);
bool server_qwen38_profile_validate_model(common_params & params, const llama_model * model, const llama_context * ctx, std::string & error);
