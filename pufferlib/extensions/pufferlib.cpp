#include <torch/extension.h>
#include <torch/torch.h>
#include <torch/optim/optimizer.h>

#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include "vecenv.h"
#include <dlfcn.h>
#include "muon.h"

#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
//#include <c10/cuda/CUDAGuard.h>

#include <nvToolsExt.h>

#include <iostream>
#include <vector>

create_environments_fn create_envs;
env_init_fn env_init;
vec_reset_fn vec_reset;
vec_step_fn vec_step;
vec_send_fn vec_send;
vec_recv_fn vec_recv;
env_close_fn env_close;
vec_close_fn vec_close;
vec_log_fn vec_log;
vec_render_fn vec_render;

torch::Dtype to_torch_dtype(int dtype) {
    if (dtype == FLOAT) {
        return torch::kFloat32;
    } else if (dtype == INT) {
        return torch::kInt32;
    } else if (dtype == UNSIGNED_CHAR) {
        return torch::kUInt8;
    } else {
        assert(false && "to_torch_dtype failed to convert dtype");
    }
    return torch::kFloat32;
}

// Torch is stupid. Had to clip out a redundant cuda sync.
void clip_grad_norm_(
    const std::vector<torch::Tensor>& parameters,
    double max_norm,
    double norm_type = 2.0
    ) {
  std::vector<torch::Tensor> params_with_grad;

  for (const auto& param : parameters) {
    auto& grad = param.grad();
    if (grad.defined()) {
      params_with_grad.push_back(param);
    }
  }

  if (params_with_grad.empty()) {
    return;
  }

  torch::Tensor total_norm_tensor;
  if (norm_type == std::numeric_limits<double>::infinity()) {
    std::vector<torch::Tensor> norms;
    norms.reserve(params_with_grad.size());

    for (const auto& param : params_with_grad) {
      norms.emplace_back(param.grad().data().abs().max());
    }
    total_norm_tensor =
        (norms.size() == 1) ? norms[0] : torch::max(torch::stack(norms));
  } else if (norm_type == 0) {
    total_norm_tensor =
        torch::full({}, static_cast<double>(params_with_grad.size()));
  } else {
    std::vector<torch::Tensor> norms;
    norms.reserve(params_with_grad.size());

    for (const auto& param : params_with_grad) {
      norms.emplace_back(param.grad().data().norm(norm_type));
    }
    total_norm_tensor =
        (norms.size() == 1) ? norms[0] : torch::stack(norms).norm(norm_type);
  }

  auto clip_coef = max_norm / (total_norm_tensor + 1e-6);
  auto clip_coef_clamped =
      torch::clamp(clip_coef, std::nullopt /* min */, 1.0 /* max */);
  for (auto& param : params_with_grad) {
    param.grad().data().mul_(clip_coef_clamped);
  }
}

std::tuple<VecEnv*, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
create_environments(int64_t num_envs, int threads) {
    void* handle = dlopen("./breakout.so", RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "dlopen error: %s\n", dlerror());
        exit(1);
    }
    dlerror();

    // Load the function pointer
    create_envs = (create_environments_fn)dlsym(handle, "create_environments");
    env_init = (env_init_fn)dlsym(handle, "env_init");
    vec_reset = (vec_reset_fn)dlsym(handle, "vec_reset");
    vec_step = (vec_step_fn)dlsym(handle, "vec_step");
    vec_send = (vec_send_fn)dlsym(handle, "vec_send");
    vec_recv = (vec_recv_fn)dlsym(handle, "vec_recv");
    env_close = (env_close_fn)dlsym(handle, "env_close");
    vec_close = (vec_close_fn)dlsym(handle, "vec_close");
    vec_log = (vec_log_fn)dlsym(handle, "vec_log");
    vec_render = (vec_render_fn)dlsym(handle, "vec_render");
    int obs_n = *(int*)dlsym(handle, "OBS_N");
    int act_n = *(int*)dlsym(handle, "ACT_N");
    int obs_t = *(int*)dlsym(handle, "OBS_T");
    int act_t = *(int*)dlsym(handle, "ACT_T");
    
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
        fprintf(stderr, "dlsym error: %s\n", dlsym_error);
        dlclose(handle);
        exit(1);
    }

    Dict* kwargs = create_dict(32);
    dict_set_int(kwargs, "frameskip", 4);
    dict_set_int(kwargs, "width", 576);
    dict_set_int(kwargs, "height", 330);
    dict_set_int(kwargs, "paddle_width", 62);
    dict_set_int(kwargs, "paddle_height", 8);
    dict_set_int(kwargs, "ball_width", 32);
    dict_set_int(kwargs, "ball_height", 32);
    dict_set_int(kwargs, "brick_width", 32);
    dict_set_int(kwargs, "brick_height", 12);
    dict_set_int(kwargs, "brick_rows", 6);
    dict_set_int(kwargs, "brick_cols", 18);
    dict_set_int(kwargs, "initial_ball_speed", 256);
    dict_set_int(kwargs, "max_ball_speed", 448);
    dict_set_int(kwargs, "paddle_speed", 620);
    dict_set_int(kwargs, "continuous", 0);

    /*
    Dict* kwargs = create_dict(32);
    dict_set_int(kwargs, "can_go_over_65536", 0);
    dict_set_float(kwargs, "reward_scaler", 0.67);
    dict_set_float(kwargs, "endgame_env_prob", 0.05);
    dict_set_float(kwargs, "scaffolding_ratio", 0.67);
    dict_set_int(kwargs, "use_heuristic_rewards", 1);
    dict_set_float(kwargs, "snake_reward_weight", 0.0005);
    dict_set_int(kwargs, "use_sparse_reward", 0);
    */

    VecEnv* vec = create_envs(num_envs, threads, 2, 256, true, 0, kwargs);
    printf("Created VecEnv with %d environments\n", vec->size);

    // Close the library
    //dlclose(handle);
 
    auto obs_dtype = to_torch_dtype(obs_t);
    auto atn_dtype = to_torch_dtype(act_t);

    auto obs = torch::from_blob(vec->gpu_observations, {num_envs, obs_n}, torch::dtype(obs_dtype).device(torch::kCUDA));
    auto actions = torch::from_blob(vec->gpu_actions, {num_envs}, torch::dtype(atn_dtype).device(torch::kCUDA));
    auto rewards = torch::from_blob(vec->gpu_rewards, {num_envs}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto terminals = torch::from_blob(vec->gpu_terminals, {num_envs}, torch::dtype(torch::kUInt8).device(torch::kCUDA));

    // TODO: RESET
    vec_reset(vec);
    return std::make_tuple(vec, obs, actions, rewards, terminals);
}

namespace py = pybind11;

// Forward declare modules
torch::Tensor mingru_gate(
    torch::Tensor state,
    torch::Tensor gate,
    torch::Tensor hidden
);
torch::autograd::tensor_list log_coeffs_and_values(
    torch::Tensor gate,
    torch::Tensor hidden
);
torch::autograd::tensor_list fused_scan(
    torch::Tensor log_coeffs,
    torch::Tensor log_values
);
torch::Tensor logcumsumexp_cuda(torch::Tensor x);
torch::autograd::tensor_list fused_ppo_loss(
    torch::Tensor logits,
    torch::Tensor values_pred,
    torch::Tensor actions,
    torch::Tensor old_logprobs,
    torch::Tensor advantages,
    torch::Tensor prio,
    torch::Tensor values,
    torch::Tensor returns,
    torch::Tensor adv_mean,
    torch::Tensor adv_std,
    float clip_coef,
    float vf_clip_coef,
    float vf_coef,
    float ent_coef
    /*
    torch::Tensor adv_mean,
    torch::Tensor adv_std,
    torch::Tensor clip_coef,
    torch::Tensor vf_clip_coef,
    torch::Tensor vf_coef,
    torch::Tensor ent_coef
    */
);

/*
torch::autograd::tensor_list rmsnorm(
    torch::Tensor x,
    torch::Tensor weight,
    double eps
);
class RMSNormImpl : public torch::nn::Module {
public:
    explicit RMSNormImpl(int64_t hidden_size, double eps = 1e-5);
    torch::Tensor forward(torch::Tensor x);
    double eps{1e-5};
    torch::Tensor weight;
};

TORCH_MODULE(RMSNorm);
*/


auto DTYPE = torch::kFloat32;

namespace pufferlib {

void puff_advantage_row(float* values, float* rewards, float* dones,
        float* importance, float* advantages, float gamma, float lambda,
        float rho_clip, float c_clip, int horizon) {
    float lastpufferlam = 0;
    for (int t = horizon-2; t >= 0; t--) {
        int t_next = t + 1;
        float nextnonterminal = 1.0 - dones[t_next];
        float rho_t = fminf(importance[t], rho_clip);
        float c_t = fminf(importance[t], c_clip);
        float delta = rho_t*(rewards[t_next] + gamma*values[t_next]*nextnonterminal - values[t]);
        lastpufferlam = delta + gamma*lambda*c_t*lastpufferlam*nextnonterminal;
        advantages[t] = lastpufferlam;
    }
}


void vtrace_check(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
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
}

// [num_steps, horizon]
void puff_advantage(float* values, float* rewards, float* dones, float* importance,
        float* advantages, float gamma, float lambda, float rho_clip, float c_clip,
        int num_steps, const int horizon){
    for (int offset = 0; offset < num_steps*horizon; offset+=horizon) {
        puff_advantage_row(values + offset, rewards + offset,
            dones + offset, importance + offset, advantages + offset,
            gamma, lambda, rho_clip, c_clip, horizon
        );
    }
}

void compute_puff_advantage_cpu(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
        double gamma, double lambda, double rho_clip, double c_clip) {
    int num_steps = values.size(0);
    int horizon = values.size(1);
    vtrace_check(values, rewards, dones, importance, advantages, num_steps, horizon);
    puff_advantage(values.data_ptr<float>(), rewards.data_ptr<float>(),
        dones.data_ptr<float>(), importance.data_ptr<float>(), advantages.data_ptr<float>(),
        gamma, lambda, rho_clip, c_clip, num_steps, horizon
    );
}

TORCH_LIBRARY(pufferlib, m) {
   m.def("compute_puff_advantage(Tensor(a!) values, Tensor(b!) rewards, Tensor(c!) dones, Tensor(d!) importance, Tensor(e!) advantages, float gamma, float lambda, float rho_clip, float c_clip) -> ()");
 }

TORCH_LIBRARY_IMPL(pufferlib, CPU, m) {
  m.impl("compute_puff_advantage", &compute_puff_advantage_cpu);
}

/*
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
create_squared_environments(int64_t num_envs, int64_t grid_size, torch::Tensor dummy);

struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

void step_environments_cuda(torch::Tensor envs_tensor, torch::Tensor indices_tensor);

void reset_environments_cuda(torch::Tensor envs_tensor, torch::Tensor indices_tensor);

Log log_environments_cuda(torch::Tensor envs_tensor, torch::Tensor indices_tensor);
*/

void compute_puff_advantage_cuda(
    torch::Tensor values,
    torch::Tensor rewards,
    torch::Tensor dones,
    torch::Tensor importance,
    torch::Tensor advantages,
    double gamma,
    double lambda,  // Note: 'lambda' is fine as a param name in C++
    double rho_clip,
    double c_clip
);

struct ShareableLSTMCell : public torch::nn::LSTMCellImpl {
    ShareableLSTMCell(const torch::nn::LSTMCellOptions& options) : torch::nn::LSTMCellImpl(options) {}

    void set_shared_weights(torch::Tensor w_ih, torch::Tensor w_hh, torch::Tensor b_ih, torch::Tensor b_hh) {
        weight_ih = w_ih;
        weight_hh = w_hh;
        bias_ih = b_ih;
        bias_hh = b_hh;

        // Remove the original (unused) tensors from the parameter dict to avoid waste
        parameters_.erase("weight_ih");
        parameters_.erase("weight_hh");
        parameters_.erase("bias_ih");
        parameters_.erase("bias_hh");
    }
};

class RMSNorm : public torch::nn::Module {
private:
    int64_t dim;
    torch::Tensor weight{nullptr};

public:
    RMSNorm(int64_t dim)
        : dim(dim) {

        weight = register_parameter("weight", torch::ones(dim));
    }

    torch::Tensor forward(torch::Tensor x) {
        int ndim = x.dim();
        TORCH_CHECK(x.size(ndim - 1) == dim, "Last dimension must match expected size");
        return torch::nn::functional::normalize(
            x, torch::nn::functional::NormalizeFuncOptions().p(2.0).dim(-1).eps(0)) * weight;
        //auto mean_sq = (x*x).mean(ndim - 1, true);
        //return weight * x/mean_sq.sqrt();
    }
};


class MinGRULayer : public torch::nn::Module {
private:
    int64_t dim;
    torch::nn::Linear to_hidden_and_gate{nullptr};
    torch::nn::Linear to_out{nullptr};
    //torch::Tensor rmsnorm_weight{nullptr};
    //RMSNorm rmsnorm{nullptr};
    std::shared_ptr<RMSNorm> rmsnorm{nullptr};
    bool kernels;

public:
    int64_t expansion_factor;
    MinGRULayer(int64_t dim, int64_t expansion_factor = 1., bool kernels = true)
        : dim(dim), expansion_factor(expansion_factor), kernels(kernels) {

        int dim_inner = int(dim * expansion_factor);
        to_hidden_and_gate = register_module("to_hidden_and_gate",
                torch::nn::Linear(torch::nn::LinearOptions(dim, 2*dim_inner).bias(false)));
        torch::nn::init::orthogonal_(to_hidden_and_gate->weight);

        // TODO: Is there a way to have this be identity to keep param count correct?
        //if (expansion_factor != 1.) 
        to_out = register_module("to_out",
                torch::nn::Linear(torch::nn::LinearOptions(dim*expansion_factor, dim).bias(false)));
        torch::nn::init::orthogonal_(to_out->weight);
        rmsnorm = register_module("rmsnorm", std::make_shared<RMSNorm>(dim));

        //rmsnorm_weight = register_parameter("rmsnorm_weight", torch::ones({dim}));
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x, torch::Tensor state = torch::Tensor()) {
        TORCH_CHECK(x.dim() == 3, "x must be [B, seq, input_size]");
        TORCH_CHECK(state.dim() == 3, "state must be [B, seq, hidden_size]");
        TORCH_CHECK(x.size(0) == state.size(0), "x and state must have the same batch size");

        auto seq_len = x.size(1);
        auto output = to_hidden_and_gate->forward(x);
        auto chunks = output.chunk(2, 2);
        auto hidden = chunks[0];
        auto gate = chunks[1];

        torch::Tensor out;
        torch::Tensor next_prev_hidden;

        if (seq_len == 1) {
            if (kernels) {
                out = mingru_gate(state, gate.contiguous(), hidden.contiguous());
            } else {
                hidden = torch::where(hidden >= 0, hidden + 0.5, hidden.sigmoid());
                gate = gate.sigmoid();
                out = torch::lerp(state, hidden, gate);
            }
            next_prev_hidden = out;
        } else {
            torch::Tensor log_coeffs, log_values;
            if (kernels) {
                torch::autograd::tensor_list outputs = log_coeffs_and_values(
                    gate.contiguous(), hidden.contiguous());
                log_coeffs = outputs[0];
                log_values = outputs[1];
            } else {
                log_coeffs = -torch::nn::functional::softplus(gate);
                auto log_z = -torch::nn::functional::softplus(-gate);
                auto log_tilde_h = torch::where(hidden >= 0,
                    (torch::nn::functional::relu(hidden) + 0.5).log(),
                    -torch::nn::functional::softplus(-hidden));
                log_values = log_z + log_tilde_h;
            }

            log_values = torch::cat({state.log(), log_values}, 1);
            log_coeffs = torch::pad(log_coeffs, {0, 0, 1, 0});

            // Heinsen associative scan
            if (kernels) {
                out = fused_scan(log_coeffs.contiguous(), log_values.contiguous())[0];
            } else {
                auto a_star = log_coeffs.cumsum(1);
                auto log_h0_plus_b_star = (log_values - a_star).logcumsumexp(1);
                auto log_h = a_star + log_h0_plus_b_star;
                out = log_h.exp();
            }

            out = out.narrow(1, out.size(1) - seq_len, seq_len);
            next_prev_hidden = out.narrow(1, out.size(1) - 1, 1);
        }

        if (expansion_factor != 1) {
            out = to_out->forward(out);
        }

        out = out + x;
        out = rmsnorm->forward(out);
        //out = rmsnorm(out, rmsnorm_weight, 1e-5)[0];

        return std::make_tuple(out, next_prev_hidden);
    }
};


class PolicyMinGRU : public torch::nn::Module {
private:
    torch::nn::Sequential encoder{nullptr};
    torch::nn::Linear decoder{nullptr};
    //std::shared_ptr<MinGRULayer> mingru{nullptr};
    torch::nn::ModuleList mingru{nullptr};
    bool kernels;

public:
    torch::nn::Linear value{nullptr};
    int64_t input_size;
    int64_t hidden_size;
    int64_t num_atns;
    int64_t num_layers;
    float expansion_factor;

    PolicyMinGRU(int64_t input_size, int64_t num_atns, int64_t hidden_size = 128, int64_t num_layers = 1, bool kernels = true)
        : input_size(input_size), hidden_size(hidden_size), num_atns(num_atns), num_layers(num_layers), kernels(kernels) {
        expansion_factor = 1.;
        encoder = register_module("encoder", torch::nn::Sequential(
            torch::nn::Linear(input_size, hidden_size),
            torch::nn::GELU()
        ));
        auto encoder_linear = (*encoder)[0]->as<torch::nn::LinearImpl>();
        torch::nn::init::orthogonal_(encoder_linear->weight, std::sqrt(2.0));
        torch::nn::init::constant_(encoder_linear->bias, 0.0);

        decoder = register_module("decoder", torch::nn::Linear(hidden_size, num_atns));
        torch::nn::init::orthogonal_(decoder->weight, 0.01);
        torch::nn::init::constant_(decoder->bias, 0.0);

        value = register_module("value", torch::nn::Linear(hidden_size, 1));
        torch::nn::init::orthogonal_(value->weight, 1.0);
        torch::nn::init::constant_(value->bias, 0.0);

        //mingru = register_module("mingru", std::make_shared<MinGRULayer>(hidden_size, 1));
        mingru = torch::nn::ModuleList();
        for (int64_t i = 0; i < num_layers; ++i) {
            mingru->push_back(MinGRULayer(hidden_size, 1, kernels));
        }
        register_module("mingru", mingru);
    }

    torch::Tensor initial_state(int64_t batch_size, torch::Device device) {
        return torch::zeros(
            {num_layers, batch_size, hidden_size},
            torch::dtype(torch::kFloat32).device(device)
        );
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
        torch::Tensor observations, torch::Tensor state) {
        int64_t B = observations.size(0);

        // Ensure flat input: [B, input_size]
        TORCH_CHECK(observations.dim() == 2 && observations.size(1) == input_size,
            "Observations must be [B, input_size]");

        TORCH_CHECK(state.dim() == 3 && state.size(0) == num_layers && state.size(1) == B && state.size(2) == hidden_size*expansion_factor,
            "state must be [num_layers, B, hidden_size]");

        auto hidden = encoder->forward(observations);

        hidden = hidden.unsqueeze(1);
        state = state.unsqueeze(2);

        std::tuple<torch::Tensor, torch::Tensor> mingru_out;
        std::vector<torch::Tensor> state_out;

        for (int64_t i = 0; i < num_layers; ++i) {
            auto state_in = state.select(0, i);
            auto layer = (*mingru)[i]->as<MinGRULayer>();
            mingru_out = layer->forward(hidden, state_in);
            hidden = std::get<0>(mingru_out);
            //auto state_out = std::get<1>(mingru_out);
            //state.select(0, i).copy_(state_out);
            state_out.push_back(std::get<1>(mingru_out));
        }

        hidden = hidden.squeeze(1);
        //state = state.squeeze(2);
        state = torch::stack(state_out, 0).squeeze(2);

        auto logits = decoder->forward(hidden);
        auto values = value->forward(hidden);

        return {logits, values, state};
    }

    std::tuple<torch::Tensor, torch::Tensor> forward_train(
        torch::Tensor observations, torch::Tensor state) {

        auto x = observations;
        auto x_shape = x.sizes();

        // Expecting [B, TT, input_size] or [B, input_size]
        TORCH_CHECK((x.dim() == 2 || x.dim() == 3),
                    "Observations must be [B, input_size] or [B, TT, input_size]");
        TORCH_CHECK(x.size(-1) == input_size,
                    "Last dimension of observations must match input_size");

        int64_t B = x_shape[0];
        int64_t TT = (x.dim() == 3) ? x_shape[1] : 1;

        TORCH_CHECK(state.dim() == 4 && state.size(0) == num_layers && state.size(1) == B && state.size(2) == 1 && state.size(3) == hidden_size*expansion_factor,
            "state must be [num_layers, B, 1, hidden_size]");

        // Flatten time steps if needed
        if (x.dim() == 3) {
            x = x.reshape({B * TT, input_size});
        } else {
            TT = 1;
        }

        auto hidden = encoder->forward(x);

        hidden = hidden.reshape({B, TT, hidden_size});

        std::tuple<torch::Tensor, torch::Tensor> mingru_out;
        for (int64_t i = 0; i < num_layers; ++i) {
            auto state_in = state.select(0, i);
            auto layer = (*mingru)[i]->as<MinGRULayer>();
            mingru_out = layer->forward(hidden, state_in);
            hidden = std::get<0>(mingru_out);
        }

        auto flat_hidden = hidden.reshape({-1, hidden_size});
        auto logits = decoder->forward(flat_hidden);
        auto values = value->forward(flat_hidden);

        logits = logits.reshape({B, TT, num_atns});
        values = values.reshape({B, TT, 1});

        return {logits, values};
    }
};


class PolicyLSTM : public torch::nn::Module {
private:
    int64_t input_size_;
    int64_t hidden_size_;
    int64_t num_atns_;
    torch::nn::Sequential encoder{nullptr};
    torch::nn::Linear decoder{nullptr};
    torch::nn::Linear value{nullptr};
    torch::nn::LSTM lstm{nullptr};
    std::shared_ptr<ShareableLSTMCell> cell{nullptr};

public:
    // Constructor: input_size instead of grid_size
    PolicyLSTM(int64_t input_size, int64_t num_atns, int64_t hidden_size = 128)
        : input_size_(input_size), hidden_size_(hidden_size), num_atns_(num_atns) {
        encoder = register_module("encoder", torch::nn::Sequential(
            torch::nn::Linear(input_size_, hidden_size_),
            torch::nn::GELU()
        ));
        auto encoder_linear = (*encoder)[0]->as<torch::nn::LinearImpl>();
        torch::nn::init::orthogonal_(encoder_linear->weight, std::sqrt(2.0));
        torch::nn::init::constant_(encoder_linear->bias, 0.0);

        decoder = register_module("decoder", torch::nn::Linear(hidden_size_, num_atns_));
        torch::nn::init::orthogonal_(decoder->weight, 0.01);
        torch::nn::init::constant_(decoder->bias, 0.0);

        value = register_module("value", torch::nn::Linear(hidden_size_, 1));
        torch::nn::init::orthogonal_(value->weight, 1.0);
        torch::nn::init::constant_(value->bias, 0.0);

        lstm = register_module("lstm", torch::nn::LSTM(torch::nn::LSTMOptions(hidden_size_, hidden_size_).num_layers(1)));
        torch::nn::init::orthogonal_(lstm->named_parameters()["weight_ih_l0"], 1.0);
        torch::nn::init::orthogonal_(lstm->named_parameters()["weight_hh_l0"], 1.0);
        lstm->named_parameters()["bias_ih_l0"].data().zero_();
        lstm->named_parameters()["bias_hh_l0"].data().zero_();

        cell = register_module("cell", std::make_shared<ShareableLSTMCell>(torch::nn::LSTMCellOptions(hidden_size_, hidden_size_)));
        cell->set_shared_weights(lstm->named_parameters()["weight_ih_l0"],
             lstm->named_parameters()["weight_hh_l0"],
             lstm->named_parameters()["bias_ih_l0"],
             lstm->named_parameters()["bias_hh_l0"]);
        /*
        // Share weights between LSTM and LSTMCell. Do not register or you'll double-update during optim.
        //cell = torch::nn::LSTMCell(hidden_size_, hidden_size_);
        cell = register_module("cell", torch::nn::LSTMCell(hidden_size_, hidden_size_));
        cell->named_parameters()["weight_ih"].data() = lstm->named_parameters()["weight_ih_l0"].data();
        cell->named_parameters()["weight_hh"].data() = lstm->named_parameters()["weight_hh_l0"].data();
        cell->named_parameters()["bias_ih"].data() = lstm->named_parameters()["bias_ih_l0"].data();
        cell->named_parameters()["bias_hh"].data() = lstm->named_parameters()["bias_hh_l0"].data();
        //cell->to(torch::kCUDA);
        */
    }

    // Forward for evaluation/inference (uses LSTMCell)
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(
        torch::Tensor observations, torch::Tensor h, torch::Tensor c) {
        int64_t B = observations.size(0);

        // Ensure flat input: [B, input_size]
        TORCH_CHECK(observations.dim() == 2 && observations.size(1) == input_size_,
                    "Observations must be [B, input_size]");

        if (h.defined() && h.numel() > 0) {
            TORCH_CHECK(h.dim() == 2 && h.size(0) == B && h.size(1) == hidden_size_,
                        "h must be [B, hidden_size]");
            TORCH_CHECK(c.dim() == 2 && c.size(0) == B && c.size(1) == hidden_size_,
                        "c must be [B, hidden_size]");
        }

        auto hidden = encoder->forward(observations);

        std::tuple<torch::Tensor, torch::Tensor> cell_out;
        if (h.defined() && h.numel() > 0) {
            cell_out = cell->forward(hidden, std::make_optional(std::make_tuple(h, c)));
        } else {
            cell_out = cell->forward(hidden);
        }

        auto hidden_out = std::get<0>(cell_out);
        auto c_out = std::get<1>(cell_out);

        //std::std::cout << std::fixed << std::setprecision(10);
        //std::std::cout << "Hidden 0 cpp: " << hidden_out[0][0].item<float>() << std::std::endl;


        auto logits = decoder->forward(hidden_out);
        auto values = value->forward(hidden_out);

        return {logits, values, hidden_out, c_out};
    }

    // Forward for training (uses LSTM)
    std::tuple<torch::Tensor, torch::Tensor> forward_train(
        torch::Tensor observations, torch::Tensor lstm_h, torch::Tensor lstm_c) {
        auto x = observations;
        auto x_shape = x.sizes();

        // Expecting [B, TT, input_size] or [B, input_size]
        TORCH_CHECK((x.dim() == 2 || x.dim() == 3),
                    "Observations must be [B, input_size] or [B, TT, input_size]");
        TORCH_CHECK(x.size(-1) == input_size_,
                    "Last dimension of observations must match input_size");

        int64_t B = x_shape[0];
        int64_t TT = (x.dim() == 3) ? x_shape[1] : 1;

        if (lstm_h.defined() && lstm_h.numel() > 0) {
            TORCH_CHECK(lstm_h.dim() == 3 && lstm_h.size(0) == 1 && lstm_h.size(1) == B,
                        "lstm_h must be [1, B, hidden_size]");
            TORCH_CHECK(lstm_c.dim() == 3 && lstm_c.size(0) == 1 && lstm_c.size(1) == B,
                        "lstm_c must be [1, B, hidden_size]");
        }

        // Flatten time steps if needed
        if (x.dim() == 3) {
            x = x.reshape({B * TT, input_size_});
        } else {
            TT = 1;
        }

        auto hidden = encoder->forward(x);

        hidden = hidden.reshape({B, TT, hidden_size_});
        hidden = hidden.transpose(0, 1);  // [TT, B, hidden_size]

        std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> lstm_out;
        if (lstm_h.defined() && lstm_h.numel() > 0) {
            lstm_out = lstm->forward(hidden, std::make_optional(std::make_tuple(lstm_h, lstm_c)));
        } else {
            lstm_out = lstm->forward(hidden);
        }

        hidden = std::get<0>(lstm_out);
        hidden = hidden.transpose(0, 1);  // [B, TT, hidden_size]

        auto flat_hidden = hidden.reshape({-1, hidden_size_});
        auto logits = decoder->forward(flat_hidden);
        auto values = value->forward(flat_hidden);

        logits = logits.reshape({B, TT, num_atns_});
        values = values.reshape({B, TT, 1});

        return {logits, values};
    }
};

double cosine_annealing(double lr_base, double lr_min, int64_t t, int64_t T) {
    if (T == 0) return lr_base;  // avoid division by zero
    double ratio = static_cast<double>(t) / static_cast<double>(T);
    ratio = std::max(0.0, std::min(1.0, ratio));  // clamp to [0, 1]
    return lr_min + 0.5*(lr_base - lr_min)*(1 + std::cos(M_PI * ratio));
}

void sync_fp16_fp32(pufferlib::PolicyLSTM* policy_16, pufferlib::PolicyLSTM* policy_32) {
    auto params_32 = policy_32->parameters();
    auto params_16 = policy_16->parameters();
    for (size_t i = 0; i < params_32.size(); ++i) {
        params_16[i].copy_(params_32[i].to(torch::kFloat32));
    }
}

typedef struct {
    PolicyMinGRU* policy;
    VecEnv* vec;
    torch::optim::Muon* muon;
    torch::Tensor rollout_state;
    torch::Tensor observations;
    torch::Tensor actions;
    torch::Tensor values;
    torch::Tensor logprobs;
    torch::Tensor rewards;
    torch::Tensor terminals;
    torch::Tensor ratio;
    torch::Tensor importance;
    torch::Tensor env_obs;
    torch::Tensor env_actions;
    torch::Tensor env_rewards;
    torch::Tensor env_terminals;
    torch::Tensor graph_obs;
    torch::Tensor graph_actions;
    torch::Tensor graph_state;
    torch::Tensor graph_state_out;
    torch::Tensor graph_value;
    torch::Tensor graph_logprobs;
    torch::Tensor graph_train_mb_obs;
    torch::Tensor graph_train_mb_state;
    torch::Tensor graph_train_mb_actions;
    torch::Tensor graph_train_mb_logprobs;
    torch::Tensor graph_train_mb_advantages;
    torch::Tensor graph_train_mb_prio;
    torch::Tensor graph_train_mb_values;
    torch::Tensor graph_train_mb_returns;
    torch::Tensor graph_train_logits;
    torch::Tensor graph_train_newvalue;
    torch::Tensor temp_train_idx;
    //void* cudagraphs;
    at::cuda::CUDAGraph rollout_graph;
    at::cuda::CUDAGraph train_forward_graph;
    at::cuda::CUDAGraph rollout_copy_graphs[64][2];
    torch::Tensor obs_input;
    torch::Tensor state_input;
    torch::Tensor logits_output;
    torch::Tensor value_output;
    torch::Tensor state_output;
    bool captured;
    torch::Tensor adv_mean;
    torch::Tensor adv_std;
    int segments;
    int horizon;
    int input_size;
    int num_atns;
    int hidden_size;
    int num_layers;
    int minibatch_segments;
    double lr;
    double min_lr_ratio;
    double beta1;
    double beta2;
    double eps;
    int epoch;
    int max_epochs;
    double prio_beta0;
    double prio_alpha;
    double clip_coef;
    double vf_clip_coef;
    double gamma;
    double gae_lambda;
    double vtrace_rho_clip;
    double vtrace_c_clip;
    double vf_coef;
    double ent_coef;
    double max_grad_norm;
    bool use_rnn;
    bool anneal_lr;
    int total_minibatches;
    int num_envs;
    int accumulate_minibatches;
    bool cudagraphs;
    bool kernels;
    int i_tmp;
    int j_tmp;
} PuffeRL;

pybind11::dict log_environments(pybind11::object pufferl_obj) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    auto& vec = pufferl.vec;

    Dict* out = create_dict(32);
    vec_log(vec, out);

    pybind11::dict py_out;
    for (int i = 0; i < out->size; i++) {
        py_out[out->items[i].key] = out->items[i].float_value;
    }
    return py_out;
}

torch::Tensor initial_state(pybind11::object pufferl_obj, int64_t batch_size, torch::Device device) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    auto& policy = pufferl.policy;
    return policy->initial_state(batch_size, device);
}

void forward_call(PuffeRL* pufferl) {
    torch::NoGradGuard no_grad;

    torch::Tensor obs = pufferl->graph_obs;
    torch::Tensor state = pufferl->graph_state;
    auto* policy = pufferl->policy;
 
    auto [logits, value, state_out] = policy->forward(obs, state);

    logits = torch::nan_to_num(logits);
    auto logprobs = torch::log_softmax(logits, 1);
    auto action = at::multinomial(logprobs.exp(), 1, true).squeeze(1);
    auto logprob = logprobs.gather(1, action.unsqueeze(1)).squeeze(1);

    pufferl->graph_actions.copy_(action.to(torch::kInt32), false);
    pufferl->graph_value.copy_(value.flatten(), false);
    pufferl->graph_logprobs.copy_(logprob, false);
    //pufferl->graph_state.copy_(state_out, false);
    pufferl->graph_state_out.copy_(state_out, false);
}

void rollout_copy_call(PuffeRL* pufferl) {
    int h = pufferl->i_tmp;
    int buf = pufferl->j_tmp;
    int num_buffers = 2;
    int num_envs = 8192;
    int block_size = num_envs / num_buffers;

    auto obs_buffer = pufferl->observations;
    auto act_buffer = pufferl->actions;
    auto logprob_buffer = pufferl->logprobs;
    auto rew_buffer = pufferl->rewards;
    auto term_buffer = pufferl->terminals;
    auto val_buffer = pufferl->values;

    auto obs = pufferl->env_obs;
    auto actions = pufferl->env_actions;
    auto rewards = pufferl->env_rewards;
    auto terminals = pufferl->env_terminals;

    //buf_state.copy_(pufferl->graph_state_out, false);

    // Store with non-blocking copies
    obs_buffer.select(1, h).narrow(0, buf*block_size, block_size).copy_(pufferl->graph_obs, true);
    act_buffer.select(1, h).narrow(0, buf*block_size, block_size).copy_(pufferl->graph_actions.to(torch::kInt64), true);
    logprob_buffer.select(1, h).narrow(0, buf*block_size, block_size).copy_(pufferl->graph_logprobs.to(torch::kFloat32), true);
    val_buffer.select(1, h).narrow(0, buf*block_size, block_size).copy_(pufferl->graph_value.to(torch::kFloat32), true);

    auto rewards_batch = rewards.narrow(0, buf*block_size, block_size);
    auto rewards_clamped = torch::clamp(rewards_batch, -1.0f, 1.0f);
    rew_buffer.select(1, h).narrow(0, buf*block_size, block_size).copy_(rewards_clamped.to(torch::kFloat32), true);

    auto terminals_batch = terminals.narrow(0, buf*block_size, block_size);
    term_buffer.select(1, h).narrow(0, buf*block_size, block_size).copy_(terminals_batch.to(torch::kFloat32), true);

    actions.narrow(0, buf*block_size, block_size).copy_(pufferl->graph_actions.to(torch::kFloat32), true);
}
 
//std::tuple<torch::Tensor, torch::Tensor> train_forward_call(PuffeRL* pufferl) {
void train_forward_call(PuffeRL* pufferl) {
    torch::Tensor mb_obs = pufferl->graph_train_mb_obs;
    torch::Tensor mb_state = pufferl->graph_train_mb_state;
    torch::Tensor mb_actions = pufferl->graph_train_mb_actions;
    torch::Tensor mb_logprobs = pufferl->graph_train_mb_logprobs;
    torch::Tensor mb_advantages = pufferl->graph_train_mb_advantages;
    torch::Tensor mb_prio = pufferl->graph_train_mb_prio;
    torch::Tensor mb_values = pufferl->graph_train_mb_values;
    torch::Tensor mb_returns = pufferl->graph_train_mb_returns;
    auto minibatch_segments = pufferl->minibatch_segments;
    auto horizon = pufferl->horizon;
    auto adv_mean = pufferl->adv_mean;
    auto adv_std = pufferl->adv_std;
    auto clip_coef = pufferl->clip_coef;
    auto vf_clip_coef = pufferl->vf_clip_coef;
    auto vf_coef = pufferl->vf_coef;
    auto ent_coef = pufferl->ent_coef;

    auto* policy = pufferl->policy;

    auto [logits, newvalue] = policy->forward_train(mb_obs.to(DTYPE), mb_state);

    torch::Tensor loss;
    if (pufferl->kernels) {
        loss = fused_ppo_loss(
            logits,
            newvalue,
            mb_actions,
            mb_logprobs.to(logits.dtype()),
            mb_advantages.to(logits.dtype()),
            mb_prio.to(logits.dtype()),
            mb_values.to(logits.dtype()),
            mb_returns.to(logits.dtype()),
            adv_mean,
            adv_std,
            clip_coef,
            vf_clip_coef,
            vf_coef,
            ent_coef
        )[0];
    } else {
        // Flatten for action lookup
        auto flat_logits = logits.reshape({-1, logits.size(-1)});
        auto flat_actions = mb_actions.reshape({-1});
        auto logprobs_new = torch::log_softmax(flat_logits, 1);
        auto probs_new = logprobs_new.exp();

        // Gather logprobs for taken actions
        auto newlogprob_flat = logprobs_new.gather(1, flat_actions.unsqueeze(1)).squeeze(1);
        auto newlogprob = newlogprob_flat.reshape({minibatch_segments, horizon});
        auto entropy = - (probs_new * logprobs_new).sum(1).mean();

        // Compute ratio
        auto logratio = newlogprob - mb_logprobs;
        auto ratio_new = logratio.exp();

        // Update global ratio and values in-place (matches Python)
        // This one can be commented, doesn't matter much on breakout
        pufferl->ratio.index_copy_(0, pufferl->temp_train_idx, ratio_new.detach().squeeze(-1).to(torch::kFloat32));

        // Normalize advantages: (adv - mean) / std, then weight
        auto adv_normalized = mb_advantages;
        adv_normalized = mb_prio * (adv_normalized - adv_normalized.mean()) / (adv_normalized.std() + 1e-8);

        // Policy loss
        auto pg_loss1 = -adv_normalized * ratio_new;
        auto pg_loss2 = -adv_normalized * torch::clamp(ratio_new, 1.0 - clip_coef, 1.0 + clip_coef);
        auto pg_loss = torch::max(pg_loss1, pg_loss2).mean();

        // Value loss
        newvalue = newvalue.view(mb_returns.sizes());
        auto v_clipped = mb_values + torch::clamp(newvalue - mb_values, -vf_clip_coef, vf_clip_coef);
        auto v_loss_unclipped = (newvalue - mb_returns).pow(2);
        auto v_loss_clipped = (v_clipped - mb_returns).pow(2);
        auto v_loss = 0.5 * torch::max(v_loss_unclipped, v_loss_clipped).mean();

        // This one matters a lot even on breakout
        pufferl->values.index_copy_(0, pufferl->temp_train_idx, newvalue.detach().squeeze(-1).to(torch::kFloat32));

        // Total loss
        loss = pg_loss + vf_coef*v_loss - ent_coef*entropy;
        /*
        {
            torch::NoGradGuard no_grad;

            // Accumulate stats
            pg_sum += pg_loss.detach();
            v_sum += v_loss.detach();
            ent_sum += entropy.detach();
            total_sum += loss.detach();

            // KL and clipping diagnostics (matches Python)
            auto old_kl = (-logratio).mean();
            auto kl = ((ratio_new - 1) - logratio).mean();
            auto cf = (ratio_new - 1.0).abs().gt(clip_coef).to(torch::kFloat32).mean();
            auto imp = ratio_new.mean();

            old_approx_kl_sum += old_kl.detach();
            approx_kl_sum += kl.detach();
            clipfrac_sum += cf.detach();
            importance_sum += imp.detach();
        }
        */
    }

    loss.backward();
    clip_grad_norm_(policy->parameters(), pufferl->max_grad_norm);
    pufferl->muon->step();
    pufferl->muon->zero_grad();

    //return std::make_tuple(logits, newvalue);

    //std::cout << "call logits sizes: " << pufferl->graph_train_logits.sizes() << std::endl;
    //std::cout << "call newvalue sizes: " << pufferl->graph_train_newvalue.sizes() << std::endl;

    //pufferl->graph_train_logits.copy_(logits, false);
    //pufferl->graph_train_newvalue.copy_(newvalue, false);
}

// Capture
void pufferl_capture_graph(PuffeRL* pufferl, at::cuda::CUDAGraph* graph, void (*func)(PuffeRL*)) {
    /* Checklist for avoiding diabolical capture bugs:
     * 1. Don't start separate streams before tracing (i.e. env gpu buffers)
     * 2. Make sure input/output buffer pointers don't change
     * 3. Make sure to restore the original stream after tracing
     * 4. All custom kernels need to use the default torch stream
     * 5. Make sure you are using the torch stream fns, not the c10 ones.
     * 6. Scalars get captured by value. They cannot change between calls.
     */
    at::cuda::CUDAStream current_stream = at::cuda::getCurrentCUDAStream();

    at::cuda::CUDAStream warmup_stream = at::cuda::getStreamFromPool();
    at::cuda::setCurrentCUDAStream(warmup_stream);
    for (int i = 0; i < 10; ++i) {
        func(pufferl);
    }
    warmup_stream.synchronize();

    auto cap_stream = at::cuda::getStreamFromPool();
    at::cuda::setCurrentCUDAStream(cap_stream);
    graph->capture_begin();
    func(pufferl);
    graph->capture_end();
    cap_stream.synchronize();

    cudaDeviceSynchronize();

    at::cuda::setCurrentCUDAStream(current_stream);
}

/*
// Destroy
void pufferl_destroy_cudagraph(PuffeRL* pufferl) {
    auto* graph = static_cast<at::cuda::CUDAGraph*>(pufferl->cudagraph);
    delete graph;
    pufferl->cudagraph = nullptr;
}


void capture_forward(std::unique_ptr<pufferlib::PuffeRL>& pufferl) {
    auto& policy = pufferl->policy;
    pufferl->forward_graph.reset();
    pufferl->forward_graph.capture_begin(
        c10::cuda::MempoolId_t{0, 0},
        cudaStreamCaptureModeGlobal
    );

    try {
        auto output_tuple = policy->forward(pufferl->obs_buf, pufferl->state_in_buf);
        pufferl->logits_buf  = std::get<0>(output_tuple);
        pufferl->value_buf   = std::get<1>(output_tuple);
        pufferl->state_out_buf = std::get<2>(output_tuple);

        pufferl->forward_graph.capture_end();
    } catch (...) {
        pufferl->forward_graph.reset();
        throw;
    }

    pufferl->forward_graph.instantiate();
}
*/

std::unique_ptr<pufferlib::PuffeRL> create_pufferl(pybind11::dict kwargs) {
    auto pufferl = std::make_unique<pufferlib::PuffeRL>();

    pufferl->segments = kwargs["segments"].cast<int>();
    pufferl->horizon = kwargs["horizon"].cast<int>();
    pufferl->input_size = kwargs["input_size"].cast<int>();
    pufferl->num_atns = kwargs["num_atns"].cast<int>();
    pufferl->hidden_size = kwargs["hidden_size"].cast<int>();
    pufferl->num_layers = kwargs["num_layers"].cast<int>();
    pufferl->minibatch_segments = kwargs["minibatch_segments"].cast<int>();
    pufferl->lr = kwargs["lr"].cast<double>();
    pufferl->min_lr_ratio = kwargs["min_lr_ratio"].cast<double>();
    pufferl->beta1 = kwargs["beta1"].cast<double>();
    pufferl->beta2 = kwargs["beta2"].cast<double>();
    pufferl->eps = kwargs["eps"].cast<double>();
    pufferl->max_epochs = kwargs["max_epochs"].cast<int>();
    pufferl->prio_beta0 = kwargs["prio_beta0"].cast<double>();
    pufferl->prio_alpha = kwargs["prio_alpha"].cast<double>();
    pufferl->clip_coef = kwargs["clip_coef"].cast<double>();
    pufferl->vf_clip_coef = kwargs["vf_clip_coef"].cast<double>();
    pufferl->gamma = kwargs["gamma"].cast<double>();
    pufferl->gae_lambda = kwargs["gae_lambda"].cast<double>();
    pufferl->vtrace_rho_clip = kwargs["vtrace_rho_clip"].cast<double>();
    pufferl->vtrace_c_clip = kwargs["vtrace_c_clip"].cast<double>();
    pufferl->vf_coef = kwargs["vf_coef"].cast<double>();
    pufferl->ent_coef = kwargs["ent_coef"].cast<double>();
    pufferl->max_grad_norm = kwargs["max_grad_norm"].cast<double>();
    pufferl->use_rnn = kwargs["use_rnn"].cast<bool>();
    pufferl->anneal_lr = kwargs["anneal_lr"].cast<bool>();
    pufferl->total_minibatches = kwargs["total_minibatches"].cast<int>();
    pufferl->num_envs = kwargs["num_envs"].cast<int>();
    pufferl->accumulate_minibatches = kwargs["accumulate_minibatches"].cast<int>();
    pufferl->cudagraphs = kwargs["cudagraphs"].cast<bool>();
    pufferl->kernels = kwargs["kernels"].cast<bool>();

    // Seeding
    torch::manual_seed(42);
    torch::cuda::manual_seed(42);

    // Enable cuDNN benchmarking
    torch::globalContext().setBenchmarkCuDNN(true);
    torch::globalContext().setDeterministicCuDNN(false);
    torch::globalContext().setBenchmarkLimitCuDNN(32);

    // Enable TF32 for faster FP32 math (uses Tensor Cores on 4090)
    torch::globalContext().setAllowTF32CuBLAS(true);
    torch::globalContext().setAllowTF32CuDNN(true);

    // Enable faster FP16 reductions
    torch::globalContext().setAllowFP16ReductionCuBLAS(true);

    // BF16 reduction (if using bfloat16)
    torch::globalContext().setAllowBF16ReductionCuBLAS(true);

    int input_size = pufferl->input_size;
    int num_atns = pufferl->num_atns;
    int hidden_size = pufferl->hidden_size;
    int num_layers = pufferl->num_layers;
    bool kernels = pufferl->kernels;
    PolicyMinGRU* policy = new PolicyMinGRU(input_size, num_atns, hidden_size, num_layers, kernels);
    policy->to(torch::kCUDA);
    policy->to(DTYPE);
    pufferl->policy = policy;

    double lr = pufferl->lr;
    double beta1 = pufferl->beta1;
    double eps = pufferl->eps;
    pufferl->muon = new torch::optim::Muon(policy->parameters(),
        torch::optim::MuonOptions(lr).momentum(beta1).eps(eps));

    // Allocate buffers
    // TODO: Match env type, alloc on gpu native
    int segments = pufferl->segments;
    int horizon = pufferl->horizon;
    pufferl->observations = torch::zeros({segments, horizon, input_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    pufferl->actions = torch::zeros({segments, horizon}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    pufferl->values = torch::zeros({segments, horizon}, torch::dtype(DTYPE).device(torch::kCUDA));
    pufferl->logprobs = torch::zeros({segments, horizon}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    pufferl->rewards = torch::zeros({segments, horizon}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    pufferl->terminals = torch::zeros({segments, horizon}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    pufferl->ratio = torch::zeros({segments, horizon}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    pufferl->importance = torch::zeros({segments, horizon}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    int batch = 4096;
    pufferl->graph_obs = torch::zeros({batch, input_size}, DTYPE).to(torch::kCUDA);
    pufferl->graph_actions = torch::zeros(batch, torch::kInt32).to(torch::kCUDA);
    pufferl->graph_value = torch::zeros(batch, DTYPE).to(torch::kCUDA);
    pufferl->graph_logprobs = torch::zeros(batch, DTYPE).to(torch::kCUDA);
    pufferl->graph_state = policy->initial_state(batch, torch::kCUDA);
    pufferl->graph_state_out = policy->initial_state(batch, torch::kCUDA);
    pufferl->rollout_state = policy->initial_state(8192, torch::kCUDA);

    int minibatch_segments = pufferl->minibatch_segments;
    pufferl->graph_train_mb_obs = torch::zeros({minibatch_segments, horizon, input_size}, DTYPE).to(torch::kCUDA);
    pufferl->graph_train_mb_state = torch::zeros(
        {policy->num_layers, minibatch_segments, 1, policy->hidden_size},
        torch::dtype(DTYPE).device(torch::kCUDA)
    );

    auto options = torch::TensorOptions()
    .dtype(DTYPE)
    .device(torch::kCUDA);

    //pufferl->graph_train_logits = torch::zeros({minibatch_segments, horizon, num_atns}, options);
    //pufferl->graph_train_newvalue = torch::zeros({minibatch_segments, horizon, 1}, options);
    pufferl->graph_train_mb_actions = torch::zeros({minibatch_segments, horizon}, options).to(torch::kInt64);
    pufferl->graph_train_mb_logprobs = torch::zeros({minibatch_segments, horizon}, options);
    pufferl->graph_train_mb_advantages = torch::zeros({minibatch_segments, horizon}, options);
    pufferl->graph_train_mb_prio = torch::zeros({minibatch_segments, 1}, options);
    pufferl->graph_train_mb_values = torch::zeros({minibatch_segments, horizon}, options);
    pufferl->graph_train_mb_returns = torch::zeros({minibatch_segments, horizon}, options);
    pufferl->adv_mean = torch::zeros({1}, options);
    pufferl->adv_std = torch::ones({1}, options);

    /*
    std::cout << "value weight: " << pufferl->policy->value->weight[0][0].item<float>() << std::endl;
    {
        pybind11::gil_scoped_release no_gil;
        train_forward_call(pufferl.get());
    }
    std::cout << "value weight: " << pufferl->policy->value->weight[0][0].item<float>() << std::endl;
    */

    // FAILS IF DONE AFTER CREATE_ENVIRONMENTS
    if (pufferl->cudagraphs) {
        pufferl->rollout_graph = at::cuda::CUDAGraph();
        pufferl->train_forward_graph = at::cuda::CUDAGraph();
        pufferl_capture_graph(pufferl.get(), &pufferl->rollout_graph, forward_call);
        {
            pybind11::gil_scoped_release no_gil;
            pufferl_capture_graph(pufferl.get(), &pufferl->train_forward_graph, train_forward_call);
        }
        std::cout << "value weight: " << pufferl->policy->value->weight[0][0].item<float>() << std::endl;
    }

    auto [vec, obs, actions, rewards, terminals] = create_environments(8192, 8);
    pufferl->vec = vec;
    pufferl->env_obs = obs;
    pufferl->env_actions = actions;
    pufferl->env_rewards = rewards;
    pufferl->env_terminals = terminals;

    // TODO: stable?
    if (pufferl->cudagraphs) {
        for (int i = 0; i < 64; ++i) {
            for (int j = 0; j < 2; ++j) {
                pufferl->i_tmp = i;
                pufferl->j_tmp = j;
                pufferl->rollout_copy_graphs[i][j] = at::cuda::CUDAGraph();
                pufferl_capture_graph(pufferl.get(), &pufferl->rollout_copy_graphs[i][j], rollout_copy_call);
            }
        }
    }

    return pufferl;
}

void python_vec_recv(pybind11::object pufferl_obj, int buf) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    auto& vec = pufferl.vec;
    vec_recv(vec, buf);
}

void python_vec_send(pybind11::object pufferl_obj, int buf) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    auto& vec = pufferl.vec;
    vec_send(vec, buf);
}

torch::autograd::tensor_list env_buffers(pybind11::object pufferl_obj) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    auto& vec = pufferl.vec;
    return {pufferl.env_obs, pufferl.env_actions, pufferl.env_rewards, pufferl.env_terminals};
}

torch::Tensor rollouts(pybind11::object pufferl_obj) {
    torch::NoGradGuard no_grad;

    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    int64_t horizon = pufferl.horizon;
    int64_t num_envs = pufferl.num_envs;

    auto obs_buffer = pufferl.observations;
    auto act_buffer = pufferl.actions;
    auto logprob_buffer = pufferl.logprobs;
    auto rew_buffer = pufferl.rewards;
    auto term_buffer = pufferl.terminals;
    auto val_buffer = pufferl.values;

    auto& policy = pufferl.policy;
    auto& vec = pufferl.vec;

    auto env_obs = pufferl.env_obs;
    auto env_actions = pufferl.env_actions;
    auto env_rewards = pufferl.env_rewards;
    auto env_terminals = pufferl.env_terminals;

    auto state = pufferl.rollout_state;
    state.zero_();

    auto device = torch::kCUDA;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_buffers = 2;
    int block_size = num_envs / num_buffers;
    for (int64_t i = 0; i < num_buffers*horizon; ++i) {
        int buf = i % num_buffers;
	    int h = i / num_buffers;
        vec_recv(vec, buf);

        cudaDeviceSynchronize();
        nvtxRangePushA("rollout_copy_inputs");
        auto buf_state = state.narrow(1, buf*block_size, block_size);
        pufferl.graph_obs.copy_(pufferl.env_obs.narrow(0, buf*block_size, block_size).to(torch::kFloat32), true);
        pufferl.graph_state.copy_(buf_state, false);
        cudaDeviceSynchronize();
        nvtxRangePop();

        //cudaEventRecord(start);
        cudaDeviceSynchronize();
        nvtxRangePushA("rollout_graph");
        if (pufferl.cudagraphs) {
            pufferl.rollout_graph.replay();
        } else {
            forward_call(&pufferl);
        }
        cudaDeviceSynchronize();
        nvtxRangePop();
        //cudaEventRecord(stop);
        //cudaEventSynchronize(stop);
        
        cudaDeviceSynchronize();
        nvtxRangePushA("rollout_copy_outputs");
        buf_state.copy_(pufferl.graph_state_out, false);
        // Store with non-blocking copies
        pufferl.i_tmp = h;
        pufferl.j_tmp = buf;
        if (pufferl.cudagraphs) {
            pufferl.rollout_copy_graphs[h][buf].replay();
        } else {
            rollout_copy_call(&pufferl);
        }
        cudaDeviceSynchronize();
        nvtxRangePop();
 
        // TODO: There should be a lighter way to sync. You need to make sure the torch data streams
        // are ready because puffer vec uses different streams. Setting to non-blocking is not enough.
        cudaDeviceSynchronize();
        //c10::cuda::getCurrentCUDAStream().synchronize();

        {
            pybind11::gil_scoped_release no_gil;
            //step_environments_cuda(envs_tensor, indices_tensor);
            // Losing 1m sps here
            vec_send(vec, buf);
            //float reward_sum = 0;
            //for (int j = 0; j < vec->size; j++) {
            //    reward_sum += vec->rewards[j];
            //}
            //render_environments(envs_tensor, indices_tensor);
        }

	// Bad clamp
    //    rewards.clamp_(-1.0f, 1.0f);
    }

    return state;
}

pybind11::dict train(pybind11::object pufferl_obj) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();

    torch::Tensor observations = pufferl.observations;
    torch::Tensor actions = pufferl.actions;
    torch::Tensor logprobs = pufferl.logprobs;
    torch::Tensor rewards = pufferl.rewards;
    torch::Tensor terminals_input = pufferl.terminals;
    torch::Tensor ratio = pufferl.ratio;
    torch::Tensor values = pufferl.values;

    int64_t total_minibatches = pufferl.total_minibatches;
    int64_t minibatch_segments = pufferl.minibatch_segments;
    int64_t segments = pufferl.segments;
    int64_t accumulate_minibatches = pufferl.accumulate_minibatches;
    int64_t horizon = pufferl.horizon;
    double prio_beta0 = pufferl.prio_beta0;
    double prio_alpha = pufferl.prio_alpha;
    double clip_coef = pufferl.clip_coef;
    double vf_clip_coef = pufferl.vf_clip_coef;
    double gamma = pufferl.gamma;
    double gae_lambda = pufferl.gae_lambda;
    double vtrace_rho_clip = pufferl.vtrace_rho_clip;
    double vtrace_c_clip = pufferl.vtrace_c_clip;
    double vf_coef = pufferl.vf_coef;
    double ent_coef = pufferl.ent_coef;
    double max_grad_norm = pufferl.max_grad_norm;
    bool use_rnn = pufferl.use_rnn;
    bool anneal_lr = pufferl.anneal_lr;
    int64_t total_epochs = pufferl.max_epochs;
    int64_t current_epoch = pufferl.epoch;

    // Accumulators
    auto device = values.device();
    auto scalar_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    torch::Tensor pg_sum = torch::zeros({}, scalar_opts);
    torch::Tensor v_sum = torch::zeros({}, scalar_opts);
    torch::Tensor ent_sum = torch::zeros({}, scalar_opts);
    torch::Tensor total_sum = torch::zeros({}, scalar_opts);
    torch::Tensor old_approx_kl_sum = torch::zeros({}, scalar_opts);
    torch::Tensor approx_kl_sum = torch::zeros({}, scalar_opts);
    torch::Tensor clipfrac_sum = torch::zeros({}, scalar_opts);
    torch::Tensor importance_sum = torch::zeros({}, scalar_opts);

    {
    pybind11::gil_scoped_release no_gil;
    auto& policy = pufferl.policy;
    auto& muon = pufferl.muon;

    auto device = values.device();
    auto terminals = terminals_input.to(torch::kFloat32);

    if (anneal_lr) {
        double lr_min = pufferl.min_lr_ratio * pufferl.lr;
        double lr = cosine_annealing(pufferl.lr, lr_min,current_epoch, pufferl.max_epochs);
        muon->param_groups().at(0).options().set_lr(lr);
    }

    // Annealed priority exponent
    double anneal_beta = prio_beta0 + (1.0 - prio_beta0) * prio_alpha * static_cast<double>(current_epoch) / total_epochs;

    // Zero out ratio at start of epoch (matches Python: self.ratio[:] = 1)
    ratio.fill_(1.0);

    auto advantages = torch::zeros_like(values);
    compute_puff_advantage_cuda(
        values, rewards, terminals, ratio,
        advantages, gamma, gae_lambda,
        vtrace_rho_clip, vtrace_c_clip
    );

    pufferl.adv_mean.copy_(advantages.mean().detach());
    pufferl.adv_std.copy_(advantages.std().detach());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int64_t mb = 0; mb < total_minibatches; ++mb) {
        advantages.fill_(0.0);

        cudaDeviceSynchronize();
        nvtxRangePushA("compute_puff_advantage");
        compute_puff_advantage_cuda(
            values, rewards, terminals, ratio,
            advantages, gamma, gae_lambda,
            vtrace_rho_clip, vtrace_c_clip
        );
        cudaDeviceSynchronize();
        nvtxRangePop();

        cudaDeviceSynchronize();
        nvtxRangePushA("train_misc");
        // Prioritization
        auto adv = advantages.abs().sum(1);  // [num_envs]
        auto prio_weights = adv.pow(prio_alpha).nan_to_num_(0.0, 0.0, 0.0);
        auto prio_probs = (prio_weights + 1e-6)/(prio_weights.sum() + 1e-6);
        auto idx = at::multinomial(prio_probs, minibatch_segments, true);
        auto mb_prio = torch::pow(segments*prio_probs.index_select(0, idx).unsqueeze(1), -anneal_beta);

        // TODO: Needed by the cpp impl, not in the kernel yet but perf critical
        pufferl.temp_train_idx = idx;

        // Index into data
        torch::Tensor mb_obs = observations.index_select(0, idx);
        torch::Tensor mb_actions = actions.index_select(0, idx);
        torch::Tensor mb_logprobs = logprobs.index_select(0, idx);
        torch::Tensor mb_values = values.index_select(0, idx);
        torch::Tensor mb_advantages = advantages.index_select(0, idx);
        torch::Tensor mb_returns = mb_advantages + mb_values;

        // Reshape obs if not using RNN
        if (!use_rnn) {
            auto flat_shape = std::vector<int64_t>{-1, mb_obs.size(2), mb_obs.size(3)};
            mb_obs = mb_obs.reshape(flat_shape);
        }

        torch::Tensor mb_state = torch::zeros(
            {policy->num_layers, minibatch_segments, 1, policy->hidden_size},
            torch::dtype(DTYPE).device(values.device())
        );

        // Forward pass
        //auto [logits, newvalue] = policy->forward_train(mb_obs.to(DTYPE), mb_state);
        pufferl.graph_train_mb_obs.copy_(mb_obs, false);
        pufferl.graph_train_mb_state.copy_(mb_state, false);
        pufferl.graph_train_mb_actions.copy_(mb_actions, false);
        pufferl.graph_train_mb_logprobs.copy_(mb_logprobs, false);
        pufferl.graph_train_mb_advantages.copy_(mb_advantages, false);
        pufferl.graph_train_mb_prio.copy_(mb_prio, false);
        pufferl.graph_train_mb_values.copy_(mb_values, false);
        pufferl.graph_train_mb_returns.copy_(mb_returns, false);
        cudaDeviceSynchronize();
        nvtxRangePop();

        //auto [logits, newvalue] = train_forward_call(&pufferl);
        cudaDeviceSynchronize();
        nvtxRangePushA("train_forward_graph");
        if (pufferl.cudagraphs) {
            pufferl.train_forward_graph.replay();
        } else {
            train_forward_call(&pufferl);
        }
        cudaDeviceSynchronize();
        nvtxRangePop();

        //torch::Tensor loss = torch::zeros({1}, logits.options());
        // Gradient accumulation and step
        // ~10% overhead in this impl. Can save a ton of launches
        /*
        if ((mb + 1) % accumulate_minibatches == 0) {
            // We use our version that doesn't sync for no reason
            // 2m+ sps right here on clip + step!
            clip_grad_norm_(policy->parameters(), max_grad_norm);
            muon->step();
            muon->zero_grad();
            //pufferl.graph_train_logits.detach_();
            //pufferl.graph_train_newvalue.detach_();
        }
        */
    }
    pufferl.epoch += 1;

    // Compute explained variance at end of epoch
    auto y_true = advantages.flatten() + values.flatten();
    auto y_pred = values.flatten();
    auto var_y = y_true.var();
    //double explained_var = (var_y.abs() < 1e-8) ? NAN : (1 - (y_true - y_pred).var() / var_y).item<double>();

    }
    // Return losses (averaged)
    pybind11::dict losses;
    /*
    losses["pg_loss"] = pg_sum.item<float>() / total_minibatches;
    losses["value_loss"] = v_sum.item<float>() / total_minibatches;
    losses["entropy"] = ent_sum.item<float>() / total_minibatches;
    losses["total_loss"] = total_sum.item<float>() / total_minibatches;
    losses["old_approx_kl"] = old_approx_kl_sum.item<float>() / total_minibatches;
    losses["approx_kl"] = approx_kl_sum.item<float>() / total_minibatches;
    losses["clipfrac"] = clipfrac_sum.item<float>() / total_minibatches;
    losses["importance"] = importance_sum.item<float>() / total_minibatches;
    */
    //losses["explained_variance"] = explained_var;

    return losses;
}


// PYBIND11_MODULE with the extension name (pufferlib._C)
TORCH_LIBRARY(_C, m) {
    m.def("mingru_gate(Tensor state, Tensor gate, Tensor hidden) -> Tensor");
    m.def("log_coeffs_and_values(Tensor gate, Tensor hidden) -> (Tensor, Tensor)");
    m.def("fused_scan(Tensor log_coeffs, Tensor log_values) -> Tensor");
    m.def("fused_ppo_loss(Tensor logits, Tensor values, Tensor actions, Tensor old_logprobs, Tensor advantages, Tensor prio, Tensor values, Tensor returns, Tensor adv_mean, Tensor adv_std, float clip_coef, float vf_clip_coef, float vf_coef, float ent_coef) -> Tensor");
    m.def("policy_forward(Tensor obs, Tensor state) -> (Tensor, Tensor, Tensor)");
}

PYBIND11_MODULE(_C, m) {
    m.def("log_environments", &log_environments);
    m.def("rollouts", &rollouts);

    //m.def("evaluate_step", &evaluate_step);
    m.def("train", &train);
    m.def("logcumsumexp_cuda", &logcumsumexp_cuda);
    m.def("policy_forward", &PolicyMinGRU::forward);

    m.def("initial_state", &initial_state);

    // TODO: Why tf are these needed?
    m.def("mingru_gate", &mingru_gate);
    m.def("log_coeffs_and_values", &log_coeffs_and_values);
    m.def("fused_scan", &fused_scan);
    m.def("fused_ppo_loss", &fused_ppo_loss);
    //m.def("rmsnorm", &rmsnorm);

    m.def("python_vec_recv", &python_vec_recv);
    m.def("python_vec_send", &python_vec_send);
    m.def("env_buffers", &env_buffers);
    /*
    py::class_<RMSNorm, torch::nn::ModuleHolder<RMSNormImpl>>(m, "RMSNorm")
        .def(py::init<int64_t, double>(),
             py::arg("hidden_size"),
             py::arg("eps") = 1e-5)
        .def("forward", &RMSNorm::forward)
        .def("__call__", &RMSNorm::operator())
        .def_readwrite("weight", &RMSNormImpl::weight)
        .def_readonly("eps", &RMSNormImpl::eps);
    */


    py::class_<torch::optim::MuonOptions>(m, "MuonOptions")
        .def(py::init<double>());

    py::class_<torch::optim::MuonParamState>(m, "MuonParamState")
        .def(py::init<>());

    py::class_<torch::optim::Muon>(m, "Muon")
        .def(py::init<std::vector<torch::optim::OptimizerParamGroup>, torch::optim::MuonOptions>());

    m.def("create_pufferl", &create_pufferl);
    py::class_<pufferlib::PuffeRL, std::unique_ptr<pufferlib::PuffeRL>>(m, "PuffeRL")
        .def_readwrite("policy", &pufferlib::PuffeRL::policy)
        .def_readwrite("muon", &pufferlib::PuffeRL::muon);

    py::class_<pufferlib::PolicyLSTM, std::shared_ptr<pufferlib::PolicyLSTM>, torch::nn::Module> cls(m, "PolicyLSTM");
    cls.def(py::init<int64_t, int64_t, int64_t>());
    cls.def("forward", &pufferlib::PolicyLSTM::forward);
    cls.def("forward_train", &pufferlib::PolicyLSTM::forward_train);

    py::class_<pufferlib::PolicyMinGRU, std::shared_ptr<pufferlib::PolicyMinGRU>, torch::nn::Module> cls2(m, "PolicyMinGRU");
    cls2.def(py::init<int64_t, int64_t, int64_t>());
    cls2.def("forward", &pufferlib::PolicyMinGRU::forward);
    cls2.def("forward_train", &pufferlib::PolicyMinGRU::forward_train);
}
}
