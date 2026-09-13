#pragma once
#include <torch/extension.h>
#include <map>
#include <string>

void fwd_fused(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g,
    torch::Tensor beta, float scale, torch::Tensor out,
    torch::Tensor A_log, torch::Tensor dt_bias, float lower_bound,
    std::optional<torch::Tensor> initial_state,
    std::optional<torch::Tensor> final_state,
    std::map<std::string, torch::Tensor> const& buffers,
    std::map<std::string, int64_t> const& config
);
