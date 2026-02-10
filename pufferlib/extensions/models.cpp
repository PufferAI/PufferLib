// models.cpp - MinGRU, LSTM, and related model classes for pufferlib
// Included by pufferlib.cpp and profile_kernels.cu inside namespace pufferlib

using std::tuple;
using std::vector;
using std::shared_ptr;
namespace nn = torch::nn;
typedef torch::Tensor Tensor;

// Compile-time precision: default bf16, pass -DPRECISION_FLOAT for float32
#ifdef PRECISION_FLOAT
constexpr bool USE_BF16 = false;
constexpr torch::ScalarType PRECISION_DTYPE = torch::kFloat32;
#else
constexpr bool USE_BF16 = true;
constexpr torch::ScalarType PRECISION_DTYPE = torch::kBFloat16;
#endif

// Common tensor options
auto cuda_f32 = torch::dtype(torch::kFloat32).device(torch::kCUDA);
auto cuda_f64 = torch::dtype(torch::kFloat64).device(torch::kCUDA);
auto cuda_i32 = torch::dtype(torch::kInt32).device(torch::kCUDA);
auto cuda_i64 = torch::dtype(torch::kInt64).device(torch::kCUDA);

// Raw struct bundling decoder outputs: mean (logits for discrete) + logstd
struct Logits {
    Tensor mean;    // Discrete: logits. Continuous: action mean.
    Tensor logstd;  // Discrete: undefined. Continuous: log standard deviation.
};

// Minimal interfaces for swappable components
// Inherit from nn::Module so register_module works
struct Encoder : public nn::Module {
    virtual Tensor forward(Tensor x) = 0;
};

class DefaultEncoder : public Encoder {
    public:
        nn::Linear linear{nullptr};
        int input;
        int hidden;

    DefaultEncoder(int input, int hidden)
        : input(input), hidden(hidden) {
        linear = register_module("linear", nn::Linear(
            nn::LinearOptions(input, hidden).bias(false)));
        nn::init::orthogonal_(linear->weight, std::sqrt(2.0));
    }

    Tensor forward(Tensor x) override {
        return linear->forward(x.to(linear->weight.dtype()));
    }
};

struct Decoder : public nn::Module {
    virtual tuple<Logits, Tensor> forward(Tensor hidden) = 0;
};

class DefaultDecoder : public Decoder {
    public:
        nn::Linear linear{nullptr};
        Tensor logstd_param{nullptr};
        int hidden;
        int output;
        bool continuous;

    DefaultDecoder(int hidden, int output, bool continuous = false)
            : hidden(hidden), output(output), continuous(continuous) {
        linear = register_module("linear", nn::Linear(
            nn::LinearOptions(hidden, output+1).bias(false)));
        nn::init::orthogonal_(linear->weight, 0.01);
        if (continuous) {
            logstd_param = register_parameter("logstd", torch::zeros({1, output}));
        }
    }

    tuple<Logits, Tensor> forward(Tensor h) override {
        h = linear->forward(h);
        // Logits and value are fused in contiguous memory
        // This is mandatory in custom decoders for our loss kernel to work
        Logits logits = {.mean = h.narrow(-1, 0, output)};
        Tensor value = h.narrow(-1, output, 1).squeeze(-1);
        if (continuous) {
            logits.logstd = logstd_param.expand_as(logits.mean);
        }
        return {logits, value};
    }
};

// Reference implementation for mingru_gate (inference path)
// Takes combined (B, 3*H) = [hidden, gate, proj] and state (B, H)
// Returns {out, next_state} where:
//   out (B, H) = sigmoid(proj) * mingru_out
//   next_state (B, H) = mingru_out (for recurrence)
vector<Tensor> mingru_gate_cpp(Tensor state, Tensor combined) {
    auto chunks = combined.chunk(3, 1);
    auto hidden = chunks[0];
    auto gate = chunks[1];
    auto proj = chunks[2];

    auto h = torch::where(hidden >= 0, hidden + 0.5, hidden.sigmoid());
    auto g = gate.sigmoid();
    auto mingru_out = torch::lerp(state, h, g);
    auto out = torch::sigmoid(proj) * mingru_out;
    return {out, mingru_out};
}

// Reference implementation for fused_scan (training path)
// Takes combined (B, T, 3*H) = [hidden, gate, proj] and state (B, 1, H)
// Returns {out, next_state} where:
//   out (B, T, H) = sigmoid(proj) * scan_result
//   next_state (B, 1, H) = raw scan_result at T (for recurrence)
vector<Tensor> fused_scan_cpp(Tensor combined, Tensor state) {
    auto seq_len = combined.size(1);

    // Split combined into hidden, gate, proj
    auto chunks = combined.chunk(3, 2);
    auto hidden = chunks[0];
    auto gate = chunks[1];
    auto proj = chunks[2];

    // Compute log_coeffs and log_values
    auto log_coeffs = -nn::functional::softplus(gate);
    auto log_z = -nn::functional::softplus(-gate);
    auto log_tilde_h = torch::where(hidden >= 0,
        (nn::functional::relu(hidden) + 0.5).log(),
        -nn::functional::softplus(-hidden));
    auto log_values = log_z + log_tilde_h;

    // Cat state and pad for scan
    log_values = torch::cat({state.log(), log_values}, 1);
    log_coeffs = torch::pad(log_coeffs, {0, 0, 1, 0});

    // Heinsen associative scan
    auto a_star = log_coeffs.cumsum(1);
    auto log_h0_plus_b_star = (log_values - a_star).logcumsumexp(1);
    auto log_h = a_star + log_h0_plus_b_star;
    auto scan_result = log_h.exp();

    // Extract output and next_state
    scan_result = scan_result.narrow(1, scan_result.size(1) - seq_len, seq_len);
    auto next_state = scan_result.narrow(1, scan_result.size(1) - 1, 1);

    // Apply sigmoid(proj) * scan_result for output
    auto out = torch::sigmoid(proj) * scan_result;

    return {out, next_state};
}

struct RNN : public nn::Module {
    virtual tuple<Tensor, Tensor> forward(Tensor x, Tensor state) = 0;
    virtual Tensor forward_train(Tensor x, Tensor state) = 0;
    virtual Tensor initial_state(int batch_size, torch::Device device, torch::Dtype dtype) = 0;
};

struct MinGRU : public RNN {
    int hidden, num_layers;
    bool kernels;
    vector<nn::Linear> layers;

    MinGRU(int hidden, int num_layers = 1, bool kernels = true)
            : hidden(hidden), num_layers(num_layers), kernels(kernels) {
        for (int i = 0; i < num_layers; i++) {
            nn::Linear layer = nn::Linear(nn::LinearOptions(hidden, 3*hidden).bias(false));
            nn::init::orthogonal_(layer->weight);
            layers.push_back(register_module("layer_" + std::to_string(i), layer));
        }
    }

    Tensor initial_state(int batch_size, torch::Device device, torch::Dtype dtype) override {
        return torch::zeros({num_layers, batch_size, hidden},
            torch::dtype(dtype).device(device));
    }

    // Inference: x (B, H), state (num_layers, B, H) -> (h, state)
    tuple<Tensor, Tensor> forward(Tensor x, Tensor state) override {
        TORCH_CHECK(x.dim() == 2 && state.dim() == 3 && state.size(0) == num_layers
            && x.size(0) == state.size(1) && x.size(1) == hidden && state.size(2) == hidden,
            "Expected x={B, H=", hidden, "}, state={layers=", num_layers, ", B, H=", hidden, "}, ",
            "Got x=", x.sizes(), ", state=", state.sizes());

        for (int i = 0; i < num_layers; i++) {
            Tensor state_i = state.select(0, i);
            Tensor combined = layers[i]->forward(x);
            auto result = kernels ? mingru_gate(state_i, combined.contiguous())
                                  : mingru_gate_cpp(state_i, combined);
            x = result[0];
            state.select(0, i).copy_(result[1]);
        }
        return {x, state};
    }

    // Training: x (B, TT, H), state (num_layers, B, 1, H) -> h (B, TT, H)
    Tensor forward_train(Tensor x, Tensor state) override {
        TORCH_CHECK(x.dim() == 3 && x.size(2) == hidden
            && state.dim() == 4 && state.size(0) == num_layers && x.size(0) == state.size(1)
            && state.size(2) == 1 && state.size(3) == hidden,
            "Expected x={B, TT, H=", hidden, "}, state={layers=", num_layers, ", B, 1, H=", hidden, "}, ",
            "Got x=", x.sizes(), ", state=", state.sizes());

        for (int i = 0; i < num_layers; i++) {
            Tensor state_i = state.select(0, i);
            Tensor combined = layers[i]->forward(x);
            auto result = kernels ? fused_scan_checkpointed(combined, state_i)
                                  : fused_scan_cpp(combined, state_i);
            x = result[0];
        }
        return x;
    }
};

struct Policy : public nn::Module {
    int input, hidden, num_atns;
    shared_ptr<Encoder> encoder{nullptr};
    shared_ptr<Decoder> decoder{nullptr};
    shared_ptr<RNN> rnn{nullptr};

    Policy(shared_ptr<Encoder> enc, shared_ptr<Decoder> dec, shared_ptr<RNN> rnn_module,
            int input, int num_atns, int hidden = 128)
            : input(input), hidden(hidden), num_atns(num_atns) {
        encoder = register_module("encoder", enc);
        decoder = register_module("decoder", dec);
        rnn = register_module("rnn", rnn_module);
    }

    Tensor initial_state(int batch_size, torch::Device device) {
        auto dtype = parameters().empty() ? torch::kFloat32 : parameters()[0].scalar_type();
        return rnn->initial_state(batch_size, device, dtype);
    }

    tuple<Logits, Tensor, Tensor> forward(Tensor observations, Tensor state) {
        TORCH_CHECK(observations.dim() == 2 && observations.size(1) == input,
            "Expected obs={B, ", input, "}, Got ", observations.sizes());

        Tensor h = encoder->forward(observations);
        auto [h_out, state_out] = rnn->forward(h, state);
        auto [logits, values] = decoder->forward(h_out);
        return {logits, values, state_out};
    }

    tuple<Logits, Tensor> forward_train(Tensor x, Tensor state) {
        TORCH_CHECK(x.dim() == 3 && x.size(-1) == input,
            "Expected obs={B, TT, ", input, "}, Got ", x.sizes());

        int B = x.size(0);
        int TT = x.size(1);

        x = x.reshape({B*TT, input});
        Tensor h = encoder->forward(x);
        h = h.reshape({B, TT, hidden});

        h = rnn->forward_train(h, state);
        Tensor flat_h = h.reshape({-1, hidden});

        auto [logits, values] = decoder->forward(flat_h);

        logits.mean = logits.mean.reshape({B, TT, num_atns});
        if (logits.logstd.defined()) {
            logits.logstd = logits.logstd.reshape({B, TT, num_atns});
        }
        values = values.reshape({B, TT, 1});

        return {logits, values};
    }
};

struct ShareableLSTMCell : public nn::LSTMCellImpl {
    ShareableLSTMCell(const nn::LSTMCellOptions& options) : nn::LSTMCellImpl(options) {}

    void set_shared_weights(Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh) {
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

class PolicyLSTM : public nn::Module {
private:
    int input_;
    int hidden_;
    int num_atns_;
    nn::Sequential encoder{nullptr};
    nn::Linear decoder{nullptr};
    nn::Linear value{nullptr};
    nn::LSTM lstm{nullptr};
    shared_ptr<ShareableLSTMCell> cell{nullptr};

public:
    PolicyLSTM(int input, int num_atns, int hidden = 128)
        : input_(input), hidden_(hidden), num_atns_(num_atns) {
        encoder = register_module("encoder", nn::Sequential(
            nn::Linear(input_, hidden_),
            nn::GELU()
        ));
        auto encoder_linear = (*encoder)[0]->as<nn::LinearImpl>();
        nn::init::orthogonal_(encoder_linear->weight, std::sqrt(2.0));
        nn::init::constant_(encoder_linear->bias, 0.0);

        decoder = register_module("decoder", nn::Linear(hidden_, num_atns_));
        nn::init::orthogonal_(decoder->weight, 0.01);
        nn::init::constant_(decoder->bias, 0.0);

        value = register_module("value", nn::Linear(hidden_, 1));
        nn::init::orthogonal_(value->weight, 1.0);
        nn::init::constant_(value->bias, 0.0);

        lstm = register_module("lstm", nn::LSTM(nn::LSTMOptions(hidden_, hidden_).num_layers(1)));
        nn::init::orthogonal_(lstm->named_parameters()["weight_ih_l0"], 1.0);
        nn::init::orthogonal_(lstm->named_parameters()["weight_hh_l0"], 1.0);
        lstm->named_parameters()["bias_ih_l0"].data().zero_();
        lstm->named_parameters()["bias_hh_l0"].data().zero_();

        cell = register_module("cell", std::make_shared<ShareableLSTMCell>(
            nn::LSTMCellOptions(hidden_, hidden_)));
        cell->set_shared_weights(lstm->named_parameters()["weight_ih_l0"],
            lstm->named_parameters()["weight_hh_l0"],
            lstm->named_parameters()["bias_ih_l0"],
            lstm->named_parameters()["bias_hh_l0"]);
    }

    // Forward for evaluation/inference (uses LSTMCell)
    tuple<Tensor, Tensor, Tensor, Tensor> forward(
        Tensor observations, Tensor h, Tensor c) {
        int64_t B = observations.size(0);

        TORCH_CHECK(observations.dim() == 2 && observations.size(1) == input_,
                    "Observations must be [B, input]");

        if (h.defined() && h.numel() > 0) {
            TORCH_CHECK(h.dim() == 2 && h.size(0) == B && h.size(1) == hidden_,
                        "h must be [B, hidden]");
            TORCH_CHECK(c.dim() == 2 && c.size(0) == B && c.size(1) == hidden_,
                        "c must be [B, hidden]");
        }

        Tensor hidden = encoder->forward(observations);

        tuple<Tensor, Tensor> cell_out;
        if (h.defined() && h.numel() > 0) {
            cell_out = cell->forward(hidden, std::make_optional(std::make_tuple(h, c)));
        } else {
            cell_out = cell->forward(hidden);
        }

        Tensor hidden_out = std::get<0>(cell_out);
        Tensor c_out = std::get<1>(cell_out);

        Tensor logits = decoder->forward(hidden_out);
        Tensor values = value->forward(hidden_out);

        return {logits, values, hidden_out, c_out};
    }

    // Forward for training (uses LSTM)
    tuple<Tensor, Tensor> forward_train(
        Tensor observations, Tensor lstm_h, Tensor lstm_c) {
        Tensor x = observations;
        auto x_shape = x.sizes();

        TORCH_CHECK((x.dim() == 2 || x.dim() == 3),
                    "Observations must be [B, input] or [B, TT, input]");
        TORCH_CHECK(x.size(-1) == input_,
                    "Last dimension of observations must match input");

        int64_t B = x_shape[0];
        int64_t TT = (x.dim() == 3) ? x_shape[1] : 1;

        if (lstm_h.defined() && lstm_h.numel() > 0) {
            TORCH_CHECK(lstm_h.dim() == 3 && lstm_h.size(0) == 1 && lstm_h.size(1) == B,
                        "lstm_h must be [1, B, hidden]");
            TORCH_CHECK(lstm_c.dim() == 3 && lstm_c.size(0) == 1 && lstm_c.size(1) == B,
                        "lstm_c must be [1, B, hidden]");
        }

        // Flatten time steps if needed
        if (x.dim() == 3) {
            x = x.reshape({B * TT, input_});
        } else {
            TT = 1;
        }

        Tensor hidden = encoder->forward(x);

        hidden = hidden.reshape({B, TT, hidden_});
        hidden = hidden.transpose(0, 1);  // [TT, B, hidden]

        tuple<Tensor, tuple<Tensor, Tensor>> lstm_out;
        if (lstm_h.defined() && lstm_h.numel() > 0) {
            lstm_out = lstm->forward(hidden, std::make_optional(std::make_tuple(lstm_h, lstm_c)));
        } else {
            lstm_out = lstm->forward(hidden);
        }

        hidden = std::get<0>(lstm_out);
        hidden = hidden.transpose(0, 1);  // [B, TT, hidden]

        Tensor flat_hidden = hidden.reshape({-1, hidden_});
        Tensor logits = decoder->forward(flat_hidden);
        Tensor values = value->forward(flat_hidden);

        logits = logits.reshape({B, TT, num_atns_});
        values = values.reshape({B, TT, 1});

        return {logits, values};
    }
};


void sync_fp16_fp32(PolicyLSTM* policy_16, PolicyLSTM* policy_32) {
    auto params_32 = policy_32->parameters();
    auto params_16 = policy_16->parameters();
    for (size_t i = 0; i < params_32.size(); ++i) {
        params_16[i].copy_(params_32[i].to(torch::kFloat32));
    }
}

// Sync bf16 working weights from fp32 master weights (for mixed-precision training)
void sync_policy_weights(Policy* policy_bf16, Policy* policy_fp32) {
    auto params_fp32 = policy_fp32->parameters();
    auto params_bf16 = policy_bf16->parameters();
    for (size_t i = 0; i < params_fp32.size(); ++i) {
        params_bf16[i].data().copy_(params_fp32[i].data().to(torch::kBFloat16));
    }
}

// Copy gradients from bf16 policy to fp32 policy (for optimizer step)
void copy_gradients_to_fp32(Policy* policy_bf16, Policy* policy_fp32) {
    auto params_fp32 = policy_fp32->parameters();
    auto params_bf16 = policy_bf16->parameters();
    for (size_t i = 0; i < params_fp32.size(); ++i) {
        if (params_bf16[i].grad().defined()) {
            params_fp32[i].mutable_grad() = params_bf16[i].grad().to(torch::kFloat32);
        }
    }
}

// =============================================================================
// Reference/fallback implementations (pure PyTorch, no CUDA kernels)
// Moved from modules.cu for cleaner separation of CUDA vs torch-native code
// =============================================================================

torch::autograd::tensor_list log_coeffs_and_values_cpp(Tensor gate, Tensor hidden) {
    auto log_coeffs = -nn::functional::softplus(gate);
    auto log_z = -nn::functional::softplus(-gate);
    auto log_tilde_h = torch::where(hidden >= 0,
        (nn::functional::relu(hidden) + 0.5).log(),
        -nn::functional::softplus(-hidden));
    auto log_values = log_z + log_tilde_h;
    return {log_coeffs, log_values};
}

Tensor logcumsumexp_cpp(Tensor x) {
    return x.exp().cumsum(1).log();
}

// Sample from multi-head discrete distribution
// Returns {actions (B, heads), total_logprob (B,)}
vector<Tensor> sample_discrete_cpp(Tensor logits, Tensor act_sizes_cpu, int num_heads) {
    logits = torch::nan_to_num(logits, 1e-8, 1e-8, 1e-8);
    auto split = torch::split(logits, c10::IntArrayRef(act_sizes_cpu.data_ptr<int64_t>(), num_heads), 1);
    vector<Tensor> actions_vec, logprobs_vec;
    for (int i = 0; i < num_heads; i++) {
        auto log_probs = torch::log_softmax(split[i], 1);
        auto action = at::multinomial(log_probs.exp(), 1, true);
        actions_vec.push_back(action);
        logprobs_vec.push_back(log_probs.gather(1, action));
    }
    return {torch::cat(actions_vec, 1), torch::cat(logprobs_vec, 1).sum(1)};
}

// Sample from continuous Normal distribution
// Returns {actions (B, D), total_logprob (B,)}
vector<Tensor> sample_continuous_cpp(Tensor mean, Tensor logstd) {
    auto std = logstd.exp();
    auto actions = mean + std * torch::randn_like(mean);
    auto log_prob = -0.5 * ((actions - mean) / std).pow(2) - 0.5 * std::log(2 * M_PI) - logstd;
    return {actions, log_prob.sum(1)};
}

// Compute logprob + entropy for multi-head discrete actions
// Returns {logprob (batch,), entropy scalar}
vector<Tensor> discrete_logprob_entropy_cpp(Tensor logits, Tensor actions, Tensor act_sizes_cpu, int num_heads) {
    logits = torch::nan_to_num(logits, 1e-8, 1e-8, 1e-8);
    auto split = torch::split(logits, c10::IntArrayRef(act_sizes_cpu.data_ptr<int64_t>(), num_heads), 1);
    int batch = logits.size(0);
    vector<Tensor> logprobs_vec, entropies_vec;
    for (int h = 0; h < num_heads; h++) {
        auto log_probs = torch::log_softmax(split[h], 1);
        auto probs = log_probs.exp();
        auto head_actions = actions.select(-1, h).reshape({batch}).to(torch::kInt64);
        logprobs_vec.push_back(log_probs.gather(1, head_actions.unsqueeze(1)));
        entropies_vec.push_back(-(probs * log_probs).sum(1, true));
    }
    auto logprob = torch::cat(logprobs_vec, 1).sum(1);
    auto entropy = torch::cat(entropies_vec, 1).sum(1).mean();
    return {logprob, entropy};
}

// Compute logprob + entropy for continuous Normal actions
// Returns {logprob (batch,), entropy scalar}
vector<Tensor> continuous_logprob_entropy_cpp(Tensor mean, Tensor logstd, Tensor actions) {
    auto std = logstd.exp();
    auto normalized = (actions.to(mean.dtype()) - mean) / std;
    auto log_prob = -0.5 * normalized.pow(2) - 0.5 * std::log(2 * M_PI) - logstd;
    auto logprob = log_prob.sum(1);
    constexpr float HALF_1_PLUS_LOG_2PI = 1.4189385332046727f;
    auto entropy = (HALF_1_PLUS_LOG_2PI + logstd).sum(1).mean();
    return {logprob, entropy};
}

// PPO clipped loss with clipped value loss
Tensor ppo_loss_cpp(Tensor ratio, Tensor advantages, Tensor prio,
        Tensor newvalue, Tensor values, Tensor returns, Tensor entropy,
        float clip_coef, float vf_clip_coef, float vf_coef, float ent_coef) {
    auto adv_normalized = prio * (advantages - advantages.mean()) / (advantages.std() + 1e-8);
    auto pg_loss1 = -adv_normalized * ratio;
    auto pg_loss2 = -adv_normalized * torch::clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef);
    auto pg_loss = torch::max(pg_loss1, pg_loss2).mean();

    newvalue = newvalue.view(returns.sizes());
    auto v_clipped = values + torch::clamp(newvalue - values, -vf_clip_coef, vf_clip_coef);
    auto v_loss = 0.5 * torch::max((newvalue - returns).pow(2), (v_clipped - returns).pow(2)).mean();

    return pg_loss + vf_coef * v_loss - ent_coef * entropy;
}

// Dispatch: sample actions using kernel or cpp path, write to output buffers
void sample_actions(Logits& logits, Tensor value,
        Tensor actions_out, Tensor logprobs_out, Tensor values_out,
        Tensor act_sizes, Tensor act_sizes_cpu,
        bool is_continuous, bool kernels, uint64_t rng_seed, Tensor rng_offset) {
    if (kernels) {
        Tensor logstd = logits.logstd.defined() ? logits.logstd : Tensor();
        sample_logits(logits.mean, logstd, value, actions_out, logprobs_out,
            values_out, act_sizes, rng_seed, rng_offset);
    } else {
        vector<Tensor> result;
        if (is_continuous) {
            result = sample_continuous_cpp(logits.mean, logits.logstd);
        } else {
            result = sample_discrete_cpp(logits.mean, act_sizes_cpu, actions_out.size(1));
        }
        actions_out.copy_(result[0].to(torch::kFloat64), false);
        logprobs_out.copy_(result[1], false);
        values_out.copy_(value.flatten(), false);
    }
}

// Dispatch: compute PPO loss using kernel or cpp path
// Writes ratio and newvalue to output buffers as side effect
Tensor compute_train_loss(Logits& logits, Tensor newvalue,
        Tensor actions, Tensor old_logprobs, Tensor advantages, Tensor prio,
        Tensor values, Tensor returns,
        Tensor ratio_out, Tensor newvalue_out,
        Tensor act_sizes, Tensor act_sizes_cpu,
        int minibatch_size, int horizon,
        float clip_coef, float vf_clip_coef, float vf_coef, float ent_coef,
        bool is_continuous, bool kernels) {
    if (kernels) {
        Tensor logstd_safe = logits.logstd.defined() ? logits.logstd : torch::empty({0}, logits.mean.options());
        // TODO: Try using global (epoch-level) adv mean/std instead of per-minibatch
        auto [adv_var, adv_mean] = torch::var_mean(advantages);
        return fused_ppo_loss_optimized(
            logits.mean, logstd_safe, newvalue,
            actions, old_logprobs, advantages, prio, values, returns,
            adv_mean, adv_var,  // variance, not std - kernel does sqrtf
            ratio_out, newvalue_out,
            act_sizes, clip_coef, vf_clip_coef, vf_coef, ent_coef
        )[0];
    } else {
        int num_heads = actions.size(-1);
        int batch = minibatch_size;
        int segments = batch / horizon;

        vector<Tensor> result;
        if (is_continuous) {
            TORCH_CHECK(logits.logstd.defined() && logits.logstd.numel() > 0,
                "logstd must be defined for continuous actions");
            result = continuous_logprob_entropy_cpp(
                logits.mean.reshape({batch, -1}), logits.logstd.reshape({batch, -1}),
                actions.reshape({batch, -1}));
        } else {
            result = discrete_logprob_entropy_cpp(
                logits.mean.reshape({batch, -1}), actions, act_sizes_cpu, num_heads);
        }
        Tensor ratio = (result[0].reshape({segments, horizon}) - old_logprobs).exp();
        ratio_out.copy_(ratio, false);
        newvalue_out.copy_(newvalue.squeeze(-1), false);

        return ppo_loss_cpp(ratio, advantages, prio,
            newvalue, values, returns, result[1],
            clip_coef, vf_clip_coef, vf_coef, ent_coef);
    }
}

// Fast clip_grad_norm_ for contiguous weights
// Cats all grads for one-shot norm computation, then scales each grad
void clip_grad_norm_(
    const vector<Tensor>& parameters,
    double max_norm
    ) {
  // Collect flattened grads
  vector<Tensor> flat_grads;
  flat_grads.reserve(parameters.size());

  for (const auto& param : parameters) {
    auto& grad = param.grad();
    if (grad.defined()) {
      flat_grads.push_back(grad.flatten());
    }
  }

  if (flat_grads.empty()) {
    return;
  }

  // Single cat + norm (avoids per-param norm calls)
  Tensor all_grads = torch::cat(flat_grads);
  Tensor total_norm = all_grads.to(torch::kFloat32).norm(2);

  // Compute clip coefficient
  Tensor clip_coef = torch::clamp_max(max_norm / (total_norm + 1e-6), 1.0);

  // Scale each grad in-place
  for (const auto& param : parameters) {
    auto& grad = param.grad();
    if (grad.defined()) {
      grad.mul_(clip_coef);
    }
  }
}

float cosine_annealing(float lr_base, float lr_min, int t, int T) {
    if (T == 0) return lr_base;  // avoid division by zero
    float ratio = (float)t / (float)T;
    ratio = std::max(0.0f, std::min(1.0f, ratio));  // clamp to [0, 1]
    return lr_min + 0.5f*(lr_base - lr_min)*(1.0f + std::cos(M_PI * ratio));
}

// Reference implementation for testing
Tensor fc_relu_fc_max_cpp(
    Tensor x,      // (B, N, D_in)
    Tensor W1,     // (D_mid, D_in)
    Tensor b1,     // (D_mid)
    Tensor W2,     // (D_out, D_mid)
    Tensor b2      // (D_out)
) {
    // FC1: x @ W1.T + b1 -> (B, N, D_mid)
    auto fc1 = torch::addmm(b1, x.flatten(0, 1), W1.t()).view({x.size(0), x.size(1), -1});
    // ReLU
    auto relu_out = torch::relu(fc1);
    // FC2: relu_out @ W2.T + b2 -> (B, N, D_out)
    auto fc2 = torch::addmm(b2, relu_out.flatten(0, 1), W2.t()).view({x.size(0), x.size(1), -1});
    // Max over N dimension
    return std::get<0>(fc2.max(1));
}

// Reference implementation for testing
Tensor fc_max_cpp(Tensor x, Tensor W, Tensor b) {
    // FC: x @ W.T + b -> (B, N, D_out)
    auto fc = torch::addmm(b, x.flatten(0, 1), W.t()).view({x.size(0), x.size(1), -1});
    // Max over N dimension
    return std::get<0>(fc.max(1));
}
