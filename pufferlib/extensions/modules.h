#ifndef PUFFERLIB_MODULES_H
#define PUFFERLIB_MODULES_H

#include <torch/extension.h>
#include <torch/torch.h>

// CUDA kernel wrappers (implemented in modules.cu)

// Fused mingru gate inference
std::vector<torch::Tensor> mingru_gate(torch::Tensor state, torch::Tensor combined);

// Fused scan with checkpointing (training path)
torch::autograd::tensor_list fused_scan_checkpointed(torch::Tensor combined, torch::Tensor state);

// LogCumSumExp
torch::Tensor logcumsumexp_cuda(torch::Tensor x);

// Fused PPO loss (optimized kernel version)
torch::autograd::tensor_list fused_ppo_loss_optimized(
    torch::Tensor logits,
    torch::Tensor logstd,
    torch::Tensor values_pred,
    torch::Tensor actions,
    torch::Tensor old_logprobs,
    torch::Tensor advantages,
    torch::Tensor prio,
    torch::Tensor values,
    torch::Tensor returns,
    torch::Tensor adv_mean,
    torch::Tensor adv_var,
    torch::Tensor ratio_out,
    torch::Tensor newvalue_out,
    torch::Tensor act_sizes,
    float clip_coef,
    float vf_clip_coef,
    float vf_coef,
    float ent_coef
);

// Fused sample_logits: handles both discrete and continuous action sampling
void sample_logits(
    torch::Tensor logits,
    torch::Tensor logstd,
    torch::Tensor value,
    torch::Tensor actions_out,
    torch::Tensor logprobs_out,
    torch::Tensor value_out,
    torch::Tensor act_sizes,
    uint64_t seed,
    torch::Tensor offset
);

// FCMax: fused FC -> Max
torch::Tensor fc_max(torch::Tensor x, torch::Tensor W, torch::Tensor b);

#endif // PUFFERLIB_MODULES_H
