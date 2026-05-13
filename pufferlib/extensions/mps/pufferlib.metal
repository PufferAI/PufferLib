#include <metal_stdlib>
using namespace metal;

kernel void puff_advantage_kernel(
    device const float* values [[buffer(0)]],
    device const float* rewards [[buffer(1)]],
    device const float* dones [[buffer(2)]],
    device const float* importance [[buffer(3)]],
    device float* advantages [[buffer(4)]],
    constant float& gamma [[buffer(5)]],
    constant float& lambda [[buffer(6)]],
    constant float& rho_clip [[buffer(7)]],
    constant float& c_clip [[buffer(8)]],
    constant int& horizon [[buffer(9)]],
    uint row [[thread_position_in_grid]])
{
    int offset = row * horizon;
    device const float* row_values = values + offset;
    device const float* row_rewards = rewards + offset;
    device const float* row_dones = dones + offset;
    device const float* row_importance = importance + offset;
    device float* row_advantages = advantages + offset;

    float gamma_lambda = gamma * lambda;
    
    float lastpufferlam = 0.0f;
    for (int t = horizon - 2; t >= 0; t--) {
        int t_next = t + 1;

        float importance_t = row_importance[t];
        float done_next = row_dones[t_next];
        float value_t = row_values[t];
        float value_next = row_values[t_next];
        float reward_next = row_rewards[t_next];
        
        float rho_t = fmin(importance_t, rho_clip);
        float c_t = fmin(importance_t, c_clip);
        
        float nextnonterminal = 1.0f - done_next;
        float delta = rho_t * (reward_next + gamma * value_next * nextnonterminal - value_t);
        lastpufferlam = delta + gamma_lambda * c_t * lastpufferlam * nextnonterminal;
        row_advantages[t] = lastpufferlam;
    }
}
