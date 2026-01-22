#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace pufferlib {

void puff_advantage_row(float* values, float* rewards, float* dones, float* importance,
        float* advantages, float value_bs, float reward_bs, float done_bs, float trunc_bs,
        float gamma, float lambda, float rho_clip, float c_clip, int horizon) {
    float lastpufferlam = 0.0;
    float next_value = 0.0;
    float nextnonterminal = 1.0;
    float next_reward = 0.0;
    for (int t = horizon-1; t >= 0; t--) {
        int t_next = t + 1;
        if ((t+1) == horizon) {
            nextnonterminal = 1.0 - done_bs;
            next_value = value_bs;
            next_reward = reward_bs;
        } else {
            nextnonterminal = 1.0 - dones[t_next];
            next_value = values[t_next];
            next_reward = rewards[t_next];
        }
        float rho_t = fminf(importance[t], rho_clip);
        float c_t = fminf(importance[t], c_clip);
        float delta = rho_t*(next_reward + gamma*next_value*nextnonterminal - values[t]);
        lastpufferlam = delta + gamma*lambda*c_t*lastpufferlam*nextnonterminal;
        advantages[t] = lastpufferlam;
    }
}

void vtrace_check(torch::Tensor values, torch::Tensor rewards, torch::Tensor dones,
                  torch::Tensor importance, torch::Tensor advantages, torch::Tensor values_bs,
                  torch::Tensor rewards_bs, torch::Tensor dones_bs, torch::Tensor truncs_bs,
                  int num_steps, int horizon) {

    // Validate input tensors
    torch::Device device = values.device();
    for (const torch::Tensor& t : {values, rewards, dones, importance, advantages}) {
        TORCH_CHECK(t.dim() == 2, "Tensor must be 2D");
        TORCH_CHECK(t.device() == device, "All tensors must be on same device");
        TORCH_CHECK(t.size(0) == num_steps, "First dimension must match num_steps");
        TORCH_CHECK(t.size(1) == horizon, "Second dimension must match horizon");
        TORCH_CHECK(t.dtype() == torch::kFloat32, "All tensors must be float32");
        if (!t.is_contiguous()) {
            t.contiguous();
        }
    }
    for (const torch::Tensor& t : {values_bs, rewards_bs, dones_bs, truncs_bs}) {
        TORCH_CHECK(t.dim() == 1, "Bootstrap Tensors must be 1D");
        TORCH_CHECK(t.device() == device, "All tensors must be on same device");
        TORCH_CHECK(t.size(0) == num_steps, "First dimension must match num_steps");
        TORCH_CHECK(t.dtype() == torch::kFloat32, "All tensors must be float32");
        if (!t.is_contiguous()) {
            t.contiguous();
        }
    }
}


// [num_steps, horizon]
void puff_advantage(float* values, float* rewards, float* dones, float* importance,
        float* advantages, float* values_bs, float* rewards_bs, float* dones_bs, float* truncs_bs,
        float gamma, float lambda, float rho_clip, float c_clip, int num_steps, const int horizon){
    int idx = 0;
    for (int offset = 0; offset < num_steps*horizon; offset+=horizon) {
        puff_advantage_row(values + offset, rewards + offset,
            dones + offset, importance + offset, advantages + offset,
            values_bs[idx], rewards_bs[idx], dones_bs[idx], truncs_bs[idx],
            gamma, lambda, rho_clip, c_clip, horizon
        );
        idx++;
    }
}


void compute_puff_advantage_cpu(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
        torch::Tensor values_bs, torch::Tensor rewards_bs, torch::Tensor dones_bs,
        torch::Tensor truncs_bs, double gamma, double lambda, double rho_clip, double c_clip) {
    int num_steps = values.size(0);
    int horizon = values.size(1);
    vtrace_check(values, rewards, dones, importance, advantages, values_bs, rewards_bs,
                 dones_bs, truncs_bs, num_steps, horizon);

    puff_advantage(
        values.data_ptr<float>(),
        rewards.data_ptr<float>(),
        dones.data_ptr<float>(),
        importance.data_ptr<float>(),
        advantages.data_ptr<float>(),
        values_bs.data_ptr<float>(),
        rewards_bs.data_ptr<float>(),
        dones_bs.data_ptr<float>(),
        truncs_bs.data_ptr<float>(),
        gamma,
        lambda,
        rho_clip,
        c_clip,
        num_steps,
        horizon
    );
}

TORCH_LIBRARY(pufferlib, m) {
   m.def("compute_puff_advantage(Tensor(a!) values, Tensor(b!) rewards, Tensor(c!) dones, Tensor(d!) importance, Tensor(e!) advantages, Tensor(f!) values_bs, Tensor(g!) rewards_bs, Tensor(h!) dones_bs, Tensor(i!) truncs_bs, float gamma, float lambda, float rho_clip, float c_clip) -> ()");
}

TORCH_LIBRARY_IMPL(pufferlib, CPU, m) {
  m.impl("compute_puff_advantage", &compute_puff_advantage_cpu);
}

}
