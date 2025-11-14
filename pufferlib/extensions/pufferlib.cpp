#include <torch/extension.h>
#include <torch/torch.h>
#include <torch/optim/optimizer.h>

#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../ocean/breakout/breakout.h"
//#include "muon.h"

//#include <ATen/cuda/CUDAGraph.h>
//#include <c10/cuda/CUDAGuard.h>

#include <iostream>
#include <vector>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
create_environments(int64_t num_envs) {
    int num_obs = 118;
    auto obs_dtype = torch::kFloat32;

    auto envs_tensor = torch::zeros({static_cast<int64_t>(num_envs * sizeof(Breakout))}, torch::kUInt8);
    auto obs = torch::zeros({num_envs, num_obs}, torch::TensorOptions().dtype(obs_dtype).pinned_memory(true));
    auto actions = torch::zeros({num_envs}, torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
    auto rewards = torch::zeros({num_envs}, torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
    auto terminals = torch::zeros({num_envs}, torch::TensorOptions().dtype(torch::kUInt8).pinned_memory(true));

    Breakout* envs = reinterpret_cast<Breakout*>(envs_tensor.data_ptr<unsigned char>());
    for (int i = 0; i < num_envs; i++) {
        Breakout* env = &envs[i];
        env->frameskip = 4;
        env->width = 576;
        env->height = 330;
        env->initial_paddle_width = 62;
        env->paddle_width = 62;
        env->paddle_height = 8;
        env->ball_width = 32;
        env->ball_height = 32;
        env->brick_width = 32;
        env->brick_height = 12;
        env->brick_rows = 6;
        env->brick_cols = 18;
        env->initial_ball_speed = 256;
        env->max_ball_speed = 448;
        env->paddle_speed = 620;
        env->continuous = 0;
        init(env);
 
        env->log = {0};
        env->observations = obs.data_ptr<float>() + i*num_obs;
        env->actions = actions.data_ptr<float>() + i;
        env->rewards = rewards.data_ptr<float>() + i;
        env->terminals = terminals.data_ptr<unsigned char>() + i;
        srand(i);
        c_reset(env);
    }
    return std::make_tuple(envs_tensor, obs, actions, rewards, terminals);
}

void step_environments(torch::Tensor envs_tensor, torch::Tensor indices_tensor) {
    Breakout* envs = reinterpret_cast<Breakout*>(envs_tensor.data_ptr<unsigned char>());
    int num_envs = indices_tensor.size(0);
    for (int i = 0; i < num_envs; i++) {
        c_step(&envs[i]);
    }
}

void reset_environments(torch::Tensor envs_tensor, torch::Tensor indices_tensor) {
    Breakout* envs = reinterpret_cast<Breakout*>(envs_tensor.data_ptr<unsigned char>());
    int num_envs = indices_tensor.size(0);
    for (int i = 0; i < num_envs; i++) {
        c_reset(&envs[i]);
    }
}

void render_environments(torch::Tensor envs_tensor, torch::Tensor indices_tensor) {
    Breakout* envs = reinterpret_cast<Breakout*>(envs_tensor.data_ptr<unsigned char>());
    c_render(&envs[0]);
}

Log log_environments(torch::Tensor envs_tensor, torch::Tensor indices_tensor) {
    Breakout* envs = reinterpret_cast<Breakout*>(envs_tensor.data_ptr<unsigned char>());
    int num_envs = indices_tensor.size(0);
    Log log = {0};
    for (int i=0; i<num_envs; i++) {
        log.perf += envs[i].log.perf;
        log.score += envs[i].log.score;
        log.episode_return += envs[i].log.episode_return;
        log.episode_length += envs[i].log.episode_length;
        log.n += envs[i].log.n;
    }
    log.perf /= log.n;
    log.score /= log.n;
    log.episode_return /= log.n;
    log.episode_length /= log.n;

    for (int i = 0; i < num_envs; i++) {
        envs[i].log = {0};
    }
    return log;
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
    float adv_mean,
    float adv_std,
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

/*
static torch::jit::Module g_policy;
void set_policy(torch::Tensor serialized_policy) {
    std::string model_str(reinterpret_cast<const char*>(serialized_policy.data_ptr<uint8_t>()), serialized_policy.numel());
    std::istringstream model_stream(model_str);
    g_policy = torch::jit::load(model_stream);
    g_policy.eval();
}
*/
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

class MinGRULayer : public torch::nn::Module {
private:
    int64_t dim;
    torch::nn::Linear to_hidden_and_gate{nullptr};
    torch::nn::Linear to_out{nullptr};

public:
    int64_t expansion_factor;
    MinGRULayer(int64_t dim, int64_t expansion_factor = 1.)
        : dim(dim), expansion_factor(expansion_factor) {

        int dim_inner = int(dim * expansion_factor);
        to_hidden_and_gate = register_module("to_hidden_and_gate",
                torch::nn::Linear(torch::nn::LinearOptions(dim, 2*dim_inner).bias(false)));
        torch::nn::init::orthogonal_(to_hidden_and_gate->weight);

        // TODO: Is there a way to have this be identity to keep param count correct?
        //if (expansion_factor != 1.) {
        to_out = register_module("to_out",
                torch::nn::Linear(torch::nn::LinearOptions(dim*expansion_factor, dim).bias(false)));
        torch::nn::init::orthogonal_(to_out->weight);
        //}
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

        //next_prev_hidden = hidden;
        //out = hidden*gate + state;

        if (seq_len == 1) {
            //hidden = torch::where(hidden >= 0, hidden + 0.5, hidden.sigmoid());
            //gate = gate.sigmoid();
            //out = torch::lerp(state, hidden, gate);
            out = mingru_gate(state, gate.contiguous(), hidden.contiguous());
            next_prev_hidden = out;
        } else {
            /*
            auto log_coeffs = -torch::nn::functional::softplus(gate);
            auto log_z = -torch::nn::functional::softplus(-gate);
            auto log_tilde_h = torch::where(hidden >= 0,
                (torch::nn::functional::relu(hidden) + 0.5).log(),
                -torch::nn::functional::softplus(-hidden));
            auto log_values = log_z + log_tilde_h;
            */
            torch::autograd::tensor_list outputs = log_coeffs_and_values(gate.contiguous(), hidden.contiguous());
            auto log_coeffs = outputs[0];
            auto log_values = outputs[1];

            log_values = torch::cat({state.log(), log_values}, 1);
            log_coeffs = torch::pad(log_coeffs, {0, 0, 1, 0});

            // Heinsen associative scan
            /*
            auto a_star = log_coeffs.cumsum(1);
            auto log_h0_plus_b_star = (log_values - a_star).logcumsumexp(1);
            auto log_h = a_star + log_h0_plus_b_star;
            out = log_h.exp();
            */

            out = fused_scan(log_coeffs.contiguous(), log_values.contiguous())[0];

            out = out.narrow(1, out.size(1) - seq_len, seq_len);
            next_prev_hidden = out.narrow(1, out.size(1) - 1, 1);
        }

        if (expansion_factor == 1) {
            out = to_out->forward(out);
        }

        return std::make_tuple(out, next_prev_hidden);
    }
};


class PolicyMinGRU : public torch::nn::Module {
private:
    torch::nn::Sequential encoder{nullptr};
    torch::nn::Linear decoder{nullptr};
    torch::nn::Linear value{nullptr};
    //std::shared_ptr<MinGRULayer> mingru{nullptr};
    torch::nn::ModuleList mingru{nullptr};

public:
    int64_t input_size;
    int64_t hidden_size;
    int64_t num_atns;
    int64_t num_layers;
    float expansion_factor;

    PolicyMinGRU(int64_t input_size, int64_t num_atns, int64_t hidden_size = 128, int64_t num_layers = 1)
        : input_size(input_size), hidden_size(hidden_size), num_atns(num_atns), num_layers(num_layers) {
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
            mingru->push_back(MinGRULayer(hidden_size, 1));
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

        for (int64_t i = 0; i < num_layers; ++i) {
            auto state_in = state.select(0, i);
            auto layer = (*mingru)[i]->as<MinGRULayer>();
            mingru_out = layer->forward(hidden, state_in);
            hidden = std::get<0>(mingru_out);
            auto state_out = std::get<1>(mingru_out);
            state.select(0, i).copy_(state_out);
        }


        hidden = hidden.squeeze(1);
        state = state.squeeze(2);

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

double cosine_annealing(double lr_base, int64_t t, int64_t T) {
    if (T == 0) return lr_base;  // avoid division by zero
    double ratio = static_cast<double>(t) / static_cast<double>(T);
    ratio = std::max(0.0, std::min(1.0, ratio));  // clamp to [0, 1]
    return lr_base * 0.5 * (1 + std::cos(M_PI * ratio));
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
    torch::optim::Muon* optimizer;
    //torch::optim::Adam* optimizer;
    double lr;
    int64_t max_epochs;
    torch::Tensor obs_buf;
    torch::Tensor state_in_buf;
    torch::Tensor logits_buf;
    torch::Tensor value_buf;
    torch::Tensor state_out_buf;
    void* cudagraph;
} PuffeRL;

/*
// Create graph
void pufferl_init_cudagraph(PuffeRL* pufferl) {
    auto graph = new at::cuda::CUDAGraph();
    pufferl->cudagraph = static_cast<void*>(graph);
}

// Capture
void pufferl_capture_forward(PuffeRL* pufferl) {
    auto& obs = pufferl->obs_buf;
    auto& state = pufferl->state_in_buf;
    auto* policy = pufferl->policy;

    c10::cuda::CUDAGuard device_guard(obs.device());

    auto* graph = static_cast<at::cuda::CUDAGraph*>(pufferl->cudagraph);

    // Warm up
    for (int i = 0; i < 3; ++i) {
        auto output = policy->forward(obs.contiguous(), state.contiguous());
        torch::cuda::synchronize();
    }

    // Ensure tensors are contiguous
    auto obs_contig = obs.contiguous();
    auto state_contig = state.contiguous();

    // Begin capture
    graph->capture_begin();  // Uses current stream

    try {
        auto output_tuple = policy->forward(obs_contig, state_contig);
        pufferl->logits_buf   = std::get<0>(output_tuple);
        pufferl->value_buf    = std::get<1>(output_tuple);
        pufferl->state_out_buf = std::get<2>(output_tuple);
        graph->capture_end();
    } catch (...) {
        graph->reset();
        throw;
    }
}

// Replay
void pufferl_replay_forward(PuffeRL* pufferl) {
    auto* graph = static_cast<at::cuda::CUDAGraph*>(pufferl->cudagraph);
    graph->replay();  // Updates outputs in place
}

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

std::unique_ptr<pufferlib::PuffeRL> create_pufferl(int64_t input_size,
        int64_t num_atns, int64_t hidden_size, int64_t num_layers,
        double lr, double beta1, double beta2, double eps, int64_t max_epochs) {

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

    auto policy = new PolicyMinGRU(input_size, num_atns, hidden_size, num_layers);
    policy->to(torch::kCUDA);
    policy->to(DTYPE);

    //auto optimizer = new torch::optim::Adam(policy->parameters(), torch::optim::AdamOptions(lr).betas({beta1, beta2}).eps(eps));
    auto optimizer = new torch::optim::Muon(policy->parameters(), torch::optim::MuonOptions(lr).eps(eps));

    auto pufferl = std::make_unique<pufferlib::PuffeRL>();
    pufferl->policy = policy;
    pufferl->optimizer = optimizer;
    pufferl->lr = lr;
    pufferl->max_epochs = max_epochs;


    //pufferl->obs_buf = torch::zeros({4096, input_size}, DTYPE).to(torch::kCUDA);
    //pufferl->state_in_buf = torch::zeros({4096, 2*hidden_size}, DTYPE).to(torch::kCUDA);

    //pufferl_init_cudagraph(pufferl.get());
    //pufferl_capture_forward(pufferl.get());

    return pufferl;
}

// Updated compiled_evaluate
torch::Tensor compiled_evaluate(
    pybind11::object pufferl_obj,
    torch::Tensor envs_tensor,
    torch::Tensor indices_tensor,
    torch::Tensor obs,
    torch::Tensor actions,
    torch::Tensor rewards,
    torch::Tensor terminals,
    //torch::Tensor lstm_h,
    torch::Tensor state,
    torch::Tensor obs_buffer,
    torch::Tensor act_buffer,
    torch::Tensor logprob_buffer,
    torch::Tensor rew_buffer,
    torch::Tensor term_buffer,
    torch::Tensor val_buffer,
    int64_t horizon,
    int64_t num_envs
) {
    torch::NoGradGuard no_grad;

    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    auto& policy = pufferl.policy;

    state = state.to(DTYPE);

    auto obs_buf = pufferl.obs_buf;
    auto state_in_buf = pufferl.state_in_buf;
    auto logits_buf = pufferl.logits_buf;
    auto value_buf = pufferl.value_buf;
    auto state_out_buf = pufferl.state_out_buf;
    //auto forward_graph = pufferl.forward_graph;

    auto device = torch::kCUDA;

    for (int64_t i = 0; i < horizon; ++i) {
        /*
        obs_buf.copy_(obs.to(DTYPE));
        state_in_buf.copy_(state.to(DTYPE));
        pufferl_replay_forward(pufferl.get());
        state = pufferl->state_out_buf;
        auto logits = logits_buf;
        auto value = value_buf;
        auto state_out = state_out_buf;
        */

        //auto [logits, value, state_out] = policy->forward(obs.to(device).to(DTYPE), state);
        auto obs_cuda = obs.to(device);
        auto [logits, value, state_out] = policy->forward(obs_cuda.to(DTYPE), state);
        state = state_out;

        logits = torch::nan_to_num(logits);

        auto logprobs = torch::log_softmax(logits, 1);
        auto action = at::multinomial(logprobs.exp(), 1, true).squeeze(1).to(torch::kInt32);
        auto logprob = logprobs.gather(1, action.unsqueeze(1)).squeeze(1);

        // Store
        obs_buffer.select(1, i).copy_(obs_cuda);
        act_buffer.select(1, i).copy_(action.to(torch::kInt64));
        logprob_buffer.select(1, i).copy_(logprob.to(torch::kFloat32));
        rew_buffer.select(1, i).copy_(rewards.to(torch::kFloat32));
        term_buffer.select(1, i).copy_(terminals.to(torch::kFloat32));
        val_buffer.select(1, i).copy_(value.flatten().to(torch::kFloat32));

        actions.copy_(action.to(torch::kCPU).to(torch::kFloat32));
        {
            pybind11::gil_scoped_release no_gil;
            //step_environments_cuda(envs_tensor, indices_tensor);
            step_environments(envs_tensor, indices_tensor);
            //render_environments(envs_tensor, indices_tensor);
        }
        rewards.clamp_(-1.0f, 1.0f);
    }

    return state;
}

/*
std::tuple<torch::Tensor, torch::Tensor> evaluate_step(
    pybind11::object pufferl_obj,
    torch::Tensor envs_tensor,
    torch::Tensor indices_tensor,
    torch::Tensor obs,
    torch::Tensor actions,
    torch::Tensor rewards,
    torch::Tensor terminals,
    torch::Tensor lstm_h,
    torch::Tensor lstm_c,
    torch::Tensor obs_buffer,
    torch::Tensor act_buffer,
    torch::Tensor logprob_buffer,
    torch::Tensor rew_buffer,
    torch::Tensor term_buffer,
    torch::Tensor val_buffer,
    int64_t i
) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    auto& policy = pufferl.policy;

    torch::NoGradGuard no_grad;

    auto [logits, value, lstm_h_out, lstm_c_out] = policy->forward(obs.to(torch::kFloat32), lstm_h, lstm_c);
    lstm_h = lstm_h_out;
    lstm_c = lstm_c_out;

    auto logprobs = torch::log_softmax(logits, 1);
    auto action = at::multinomial(logprobs.exp(), 1, true).squeeze(1);
    auto logprob = logprobs.gather(1, action.unsqueeze(1)).squeeze(1);

    // Store
    obs_buffer.select(1, i).copy_(obs.to(torch::kFloat32));
    act_buffer.select(1, i).copy_(action.to(torch::kInt32));
    logprob_buffer.select(1, i).copy_(logprob.to(torch::kFloat32));
    rew_buffer.select(1, i).copy_(rewards.to(torch::kFloat32));
    term_buffer.select(1, i).copy_(terminals.to(torch::kFloat32));
    val_buffer.select(1, i).copy_(value.flatten().to(torch::kFloat32));

    actions.copy_(action);
    return std::make_tuple(lstm_h, lstm_c);
}
*/

void batched_forward(
    pybind11::object pufferl_obj,
    torch::Tensor observations,  // [num_envs, horizon, ...]
    int64_t total_minibatches,
    int64_t minibatch_segments
) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    auto& policy = pufferl.policy;
    auto device = observations.device();
    for (int64_t mb = 0; mb < total_minibatches; ++mb) {
        float rng = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        torch::Tensor mb_obs = observations.narrow(0, mb*minibatch_segments, minibatch_segments);
        torch::Tensor mb_state = torch::zeros(
            {minibatch_segments, 1, policy->hidden_size},
            DTYPE
        ).to(device);
        auto [logits, newvalue] = policy->forward_train(mb_obs.to(DTYPE)+rng, mb_state+rng);
    }
}

pybind11::dict compiled_train(
    pybind11::object pufferl_obj,
    torch::Tensor observations,  // [num_envs, horizon, ...]
    torch::Tensor actions,       // [num_envs, horizon]
    torch::Tensor logprobs,      // [num_envs, horizon]
    torch::Tensor rewards,       // [num_envs, horizon]
    torch::Tensor terminals_input, // [num_envs, horizon]
    torch::Tensor truncations,   // [num_envs, horizon] (not used in puff advantage?)
    torch::Tensor ratio,         // [num_envs, horizon]
    torch::Tensor values,        // [num_envs, horizon]
    int64_t total_minibatches,
    int64_t minibatch_segments,
    int64_t segments,
    int64_t accumulate_minibatches,
    int64_t horizon,
    double prio_beta0,
    double prio_alpha,
    double clip_coef,
    double vf_clip_coef,
    double gamma,
    double gae_lambda,
    double vtrace_rho_clip,
    double vtrace_c_clip,
    double vf_coef,
    double ent_coef,
    double max_grad_norm,
    bool use_rnn,
    bool anneal_lr,
    int64_t total_epochs,
    int64_t current_epoch
) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();

    // Accumulators
    auto device = values.device();
    torch::Tensor pg_sum = torch::zeros({}, torch::kFloat32).to(device);
    torch::Tensor v_sum = torch::zeros({}, torch::kFloat32).to(device);
    torch::Tensor ent_sum = torch::zeros({}, torch::kFloat32).to(device);
    torch::Tensor total_sum = torch::zeros({}, torch::kFloat32).to(device);
    torch::Tensor old_approx_kl_sum = torch::zeros({}, torch::kFloat32).to(device);
    torch::Tensor approx_kl_sum = torch::zeros({}, torch::kFloat32).to(device);
    torch::Tensor clipfrac_sum = torch::zeros({}, torch::kFloat32).to(device);
    torch::Tensor importance_sum = torch::zeros({}, torch::kFloat32).to(device);

    {
    pybind11::gil_scoped_release no_gil;
    auto& policy = pufferl.policy;
    auto& optimizer = pufferl.optimizer;

    auto device = values.device();
    auto terminals = terminals_input.to(torch::kFloat32);

    if (anneal_lr) {
        double lr = cosine_annealing(pufferl.lr, current_epoch, pufferl.max_epochs);
        optimizer->param_groups().at(0).options().set_lr(lr);
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
    float adv_mean = advantages.mean().item<float>();
    float adv_std = advantages.std().item<float>();

    for (int64_t mb = 0; mb < total_minibatches; ++mb) {
        //torch::Tensor mb_obs = observations.narrow(0, mb*minibatch_segments, minibatch_segments);

        advantages.fill_(0.0);
        compute_puff_advantage_cuda(
            values, rewards, terminals, ratio,
            advantages, gamma, gae_lambda,
            vtrace_rho_clip, vtrace_c_clip
        );


        // Prioritization
        auto adv = advantages.abs().sum(1);  // [num_envs]
        auto prio_weights = adv.pow(prio_alpha).nan_to_num_(0.0, 0.0, 0.0);
        auto prio_probs = (prio_weights + 1e-6)/(prio_weights.sum() + 1e-6);
        auto idx = at::multinomial(prio_probs, minibatch_segments, true);
        auto mb_prio = torch::pow(segments*prio_probs.index_select(0, idx).unsqueeze(1), -anneal_beta);

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

        /*
        torch::Tensor mb_obs = observations.narrow(0, mb*minibatch_segments, minibatch_segments);
        torch::Tensor mb_actions = actions.narrow(0, mb*minibatch_segments, minibatch_segments);
        torch::Tensor mb_logprobs = logprobs.narrow(0, mb*minibatch_segments, minibatch_segments);
        torch::Tensor mb_values = values.narrow(0, mb*minibatch_segments, minibatch_segments);
        torch::Tensor mb_advantages = advantages.narrow(0, mb*minibatch_segments, minibatch_segments);
        torch::Tensor mb_returns = mb_advantages + mb_values;
        */


        // HARDCODED LSTM SIZE 128
        // Initial LSTM states (zero or none)
        torch::Tensor mb_state= torch::zeros(
            {policy->num_layers, minibatch_segments, 1, policy->hidden_size},
            DTYPE
        ).to(values.device());

        // Forward pass
        auto [logits, newvalue] = policy->forward_train(mb_obs.to(DTYPE), mb_state);

        /*
        std::cout << "Logits dtype: " << logits.dtype() << std::endl;
        std::cout << "Values dtype: " << newvalue.dtype() << std::endl;
        std::cout << "Actions dtype: " << mb_actions.dtype() << std::endl;
        std::cout << "Logprobs dtype: " << mb_logprobs.dtype() << std::endl;
        std::cout << "Advantages dtype: " << mb_advantages.dtype() << std::endl;
        std::cout << "Prio dtype: " << mb_prio.dtype() << std::endl;
        std::cout << "Values dtype: " << mb_values.dtype() << std::endl;
        std::cout << "Returns dtype: " << mb_returns.dtype() << std::endl;

        std::cout << "Logit shape: " << logits.sizes() << std::endl;
        std::cout << "Values shape: " << newvalue.sizes() << std::endl;
        std::cout << "Actions shape: " << mb_actions.sizes() << std::endl;
        std::cout << "Logprobs shape: " << mb_logprobs.sizes() << std::endl;
        std::cout << "Advantages shape: " << mb_advantages.sizes() << std::endl;
        std::cout << "Prio shape: " << mb_prio.sizes() << std::endl;
        std::cout << "Values shape: " << mb_values.sizes() << std::endl;
        std::cout << "Returns shape: " << mb_returns.sizes() << std::endl;
        */

        int BT = logits.size(0)*logits.size(1);
        //torch::Tensor loss = torch::zeros({1}, logits.options());
        auto loss = fused_ppo_loss(
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

        /*
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

        //auto loss = -ratio_new.mean();

        // Update global ratio and values in-place (matches Python)
        //ratio.index_copy_(0, idx, ratio_new.detach().squeeze(-1).to(torch::kFloat32));

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

        //values.index_copy_(0, idx, newvalue.detach().squeeze(-1).to(torch::kFloat32));

        // Total loss
        auto loss = pg_loss + vf_coef*v_loss - ent_coef*entropy;
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

        // Backward pass
        loss.backward();

        // Gradient accumulation and step
        if ((mb + 1) % accumulate_minibatches == 0) {
            torch::nn::utils::clip_grad_norm_(policy->parameters(), max_grad_norm);
            optimizer->step();
            optimizer->zero_grad();
        }
    }

    // Compute explained variance at end of epoch
    auto y_true = advantages.flatten() + values.flatten();
    auto y_pred = values.flatten();
    auto var_y = y_true.var();
    //double explained_var = (var_y.abs() < 1e-8) ? NAN : (1 - (y_true - y_pred).var() / var_y).item<double>();

    }
    // Return losses (averaged)
    pybind11::dict losses;
    losses["pg_loss"] = pg_sum.item<float>() / total_minibatches;
    losses["value_loss"] = v_sum.item<float>() / total_minibatches;
    losses["entropy"] = ent_sum.item<float>() / total_minibatches;
    losses["total_loss"] = total_sum.item<float>() / total_minibatches;
    losses["old_approx_kl"] = old_approx_kl_sum.item<float>() / total_minibatches;
    losses["approx_kl"] = approx_kl_sum.item<float>() / total_minibatches;
    losses["clipfrac"] = clipfrac_sum.item<float>() / total_minibatches;
    losses["importance"] = importance_sum.item<float>() / total_minibatches;
    //losses["explained_variance"] = explained_var;

    return losses;
}

// PYBIND11_MODULE with the extension name (pufferlib._C)
PYBIND11_MODULE(_C, m) {
    m.def("create_environments", &create_environments);
    m.def("step_environments", &step_environments);
    m.def("reset_environments", &reset_environments);
    m.def("log_environments", &log_environments);
    /*
    m.def("step_environments", &step_environments_cuda);
    m.def("reset_environments", &reset_environments_cuda);
    m.def("log_environments", &log_environments_cuda);
    */
    m.def("compiled_evaluate", &compiled_evaluate);

    //m.def("evaluate_step", &evaluate_step);
    m.def("compiled_train", &compiled_train);
    m.def("batched_forward", &batched_forward);
    m.def("logcumsumexp_cuda", &logcumsumexp_cuda);

    // TODO: Why tf are these needed?
    m.def("mingru_gate", &mingru_gate);
    m.def("log_coeffs_and_values", &log_coeffs_and_values);
    m.def("fused_scan", &fused_scan);
    m.def("fused_ppo_loss", &fused_ppo_loss);

    py::class_<Log>(m, "Log")
    .def_readwrite("perf", &Log::perf)
    .def_readwrite("score", &Log::score)
    .def_readwrite("episode_return", &Log::episode_return)
    .def_readwrite("episode_length", &Log::episode_length)
    .def_readwrite("n", &Log::n);

    m.def("create_pufferl", &create_pufferl);
    py::class_<pufferlib::PuffeRL, std::unique_ptr<pufferlib::PuffeRL>>(m, "PuffeRL")
        .def_readwrite("policy", &pufferlib::PuffeRL::policy)
        .def_readwrite("optimizer", &pufferlib::PuffeRL::optimizer);

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
