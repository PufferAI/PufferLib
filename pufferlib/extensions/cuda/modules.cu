#include <torch/extension.h>
#include <torch/torch.h>
#include <cuda_runtime.h>

#include "kernels.cu"

#include <stdio.h>
#include <stdlib.h>

torch::Tensor mingru_gate(
    torch::Tensor state,
    torch::Tensor gate,
    torch::Tensor hidden
) {
    // Validate
    TORCH_CHECK(state.is_cuda(), "state must be on CUDA");
    TORCH_CHECK(gate.is_cuda(), "gate must be on CUDA");
    TORCH_CHECK(hidden.is_cuda(), "hidden must be on CUDA");
    TORCH_CHECK(state.dtype() == gate.dtype() && gate.dtype() == hidden.dtype(),
                "All tensors must have the same dtype");
    TORCH_CHECK(state.sizes() == gate.sizes() && gate.sizes() == hidden.sizes(),
                "All tensors must have the same shape");
    TORCH_CHECK(state.is_contiguous() && gate.is_contiguous() && hidden.is_contiguous(),
                "All tensors must be contiguous");

    auto dtype = state.dtype();
    auto device = state.device();
    auto sizes = state.sizes();
    const int N = state.numel();

    auto out = torch::empty(sizes, state.options());

    if (dtype == torch::kFloat32) {
        launch_mingru_gate_inference<float>(
            out.data_ptr<float>(),
            gate.data_ptr<float>(),
            hidden.data_ptr<float>(),
            state.data_ptr<float>(),
            N
        );
    } else if (dtype == torch::kBFloat16) {
        launch_mingru_gate_inference<at::BFloat16>(
            out.data_ptr<at::BFloat16>(),
            gate.data_ptr<at::BFloat16>(),
            hidden.data_ptr<at::BFloat16>(),
            state.data_ptr<at::BFloat16>(),
            N
        );
    } else {
        TORCH_CHECK(false,
            "Unsupported dtype. Supported dtypes are float32 and bfloat16");
    }
    return out;
}

class LogCoeffsAndValuesFunction : public torch::autograd::Function<LogCoeffsAndValuesFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor gate,
        torch::Tensor hidden
    ) {

        auto dtype = gate.dtype();
        auto log_coeffs = torch::empty_like(gate, gate.options().dtype(dtype));
        auto log_values = torch::empty_like(gate, gate.options().dtype(dtype));

        const int N = gate.numel();

        if (dtype == torch::kFloat32) {
            launch_log_coeffs_and_values<float>(
                log_coeffs.data_ptr<float>(),
                log_values.data_ptr<float>(),
                gate.data_ptr<float>(),
                hidden.data_ptr<float>(),
                N
            );
        } else if (dtype == torch::kBFloat16) {
            launch_log_coeffs_and_values<at::BFloat16>(
                log_coeffs.data_ptr<at::BFloat16>(),
                log_values.data_ptr<at::BFloat16>(),
                gate.data_ptr<at::BFloat16>(),
                hidden.data_ptr<at::BFloat16>(),
                N
            );
        } else {
            TORCH_CHECK(false, "Unsupported dtype. Supported dtypes are float32 and bfloat16");
        }

        ctx->save_for_backward({gate, hidden});
        return {log_coeffs, log_values};
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto gate = saved[0];
        auto hidden = saved[1];
        auto grad_log_values = grad_outputs[1].contiguous();
        auto grad_log_coeffs = grad_outputs[0].contiguous();

        auto grad_gate = torch::empty_like(gate);
        auto grad_hidden = torch::empty_like(hidden);

        const int N = gate.numel();
        auto dtype = gate.dtype();

        if (dtype == torch::kFloat32) {
            launch_log_coeffs_and_values_backward<float>(
                grad_gate.data_ptr<float>(),
                grad_hidden.data_ptr<float>(),
                grad_log_coeffs.data_ptr<float>(),
                grad_log_values.data_ptr<float>(),
                gate.data_ptr<float>(),
                hidden.data_ptr<float>(),
                N
            );
        } else if (dtype == torch::kBFloat16) {
            launch_log_coeffs_and_values_backward<at::BFloat16>(
                grad_gate.data_ptr<at::BFloat16>(),
                grad_hidden.data_ptr<at::BFloat16>(),
                grad_log_coeffs.data_ptr<at::BFloat16>(),
                grad_log_values.data_ptr<at::BFloat16>(),
                gate.data_ptr<at::BFloat16>(),
                hidden.data_ptr<at::BFloat16>(),
                N
            );
        } else {
            TORCH_CHECK(false, "Unsupported dtype in backward");
        }

        return {grad_gate, grad_hidden};
    }
};

torch::autograd::tensor_list log_coeffs_and_values(
    torch::Tensor gate,
    torch::Tensor hidden
) {
    return LogCoeffsAndValuesFunction::apply(gate, hidden);
}

/*
class RMSNormFunction: public torch::autograd::Function<RMSNormFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor x,
        torch::Tensor weight,
        double eps
    ) {
        TORCH_CHECK(x.is_cuda(), "x must be on CUDA");
        TORCH_CHECK(weight.is_cuda(), "weight must be on CUDA");
        TORCH_CHECK(x.dtype() == weight.dtype(), "dtypes must match");
        TORCH_CHECK(x.dim() == 3, "x must be (B, T, H)");
        TORCH_CHECK(weight.dim() == 1, "weight must be (H,)");
        TORCH_CHECK(x.size(2) == weight.size(0), "H must match");

        auto dtype = x.dtype();
        auto device = x.device();
        auto B = x.size(0);
        auto T = x.size(1);
        auto H = x.size(2);

        auto out = torch::empty({B, T, H}, x.options());

        auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        auto inv_norm = torch::empty({B, T}, options_float);

        if (dtype == torch::kFloat32) {
            launch_rmsnorm_forward<float>(
                out.data_ptr<float>(),
                inv_norm.data_ptr<float>(),
                x.data_ptr<float>(),
                weight.data_ptr<float>(),
                static_cast<double>(eps),
                static_cast<int>(T),
                static_cast<int>(H),
                static_cast<int>(B)
            );
        } else if (dtype == torch::kBFloat16) {
            launch_rmsnorm_forward<at::BFloat16>(
                out.data_ptr<at::BFloat16>(),
                inv_norm.data_ptr<float>(),
                x.data_ptr<at::BFloat16>(),
                weight.data_ptr<at::BFloat16>(),
                static_cast<double>(eps),
                static_cast<int>(T),
                static_cast<int>(H),
                static_cast<int>(B)
            );
        } else {
            TORCH_CHECK(false, "Unsupported dtype. Only float32 and bfloat16 supported.");
        }

        // TODO: don't save eps as a tensor
        //ctx->saved_data["eps"] = eps;   // store in saved_data instead
                                    
        // Save for backward
        auto eps_tensor = torch::tensor(eps);
        ctx->save_for_backward({x, weight, out, inv_norm, eps_tensor});

        return {out};
    }
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto x = saved[0].contiguous();
        auto weight = saved[1].contiguous();
        auto out = saved[2].contiguous();
        auto inv_norm = saved[3].contiguous();
        double eps = saved[4].item<double>();

        auto grad_out = grad_outputs[0].contiguous();
        auto dtype = x.dtype();

        auto B = x.size(0);
        auto T = x.size(1);
        auto H = x.size(2);

        auto grad_x = torch::empty_like(x);
        auto grad_weight = torch::empty_like(weight);
        auto grad_eps = torch::Tensor();

        if (dtype == torch::kFloat32) {
            launch_rmsnorm_backward<float>(
                grad_x.data_ptr<float>(),
                grad_weight.data_ptr<float>(),
                grad_out.data_ptr<float>(),
                inv_norm.data_ptr<float>(),
                x.data_ptr<float>(),
                weight.data_ptr<float>(),
                eps,
                static_cast<int>(T),
                static_cast<int>(H),
                static_cast<int>(B)
            );
        } else if (dtype == torch::kBFloat16) {
            launch_rmsnorm_backward<at::BFloat16>(
                grad_x.data_ptr<at::BFloat16>(),
                grad_weight.data_ptr<at::BFloat16>(),
                grad_out.data_ptr<at::BFloat16>(),
                inv_norm.data_ptr<float>(),
                x.data_ptr<at::BFloat16>(),
                weight.data_ptr<at::BFloat16>(),
                eps,
                static_cast<int>(T),
                static_cast<int>(H),
                static_cast<int>(B)
            );
        } else {
            TORCH_CHECK(false, "Unsupported dtype");
        }

        return {grad_x, grad_weight, grad_eps};
    }
};
torch::autograd::tensor_list rmsnorm(
    torch::Tensor x,
    torch::Tensor weight,
    double eps
) {
    return RMSNormFunction::apply(x, weight, eps);
}
*/

/*
class RMSNormImpl : public torch::nn::Module {
public:
    explicit RMSNormImpl(int64_t hidden_size, double eps = 1e-5)
        : eps(eps)
    {
        // weight is the learnable scale (same shape as the last dimension)
        // We register it as a parameter so it lives on the right device and is trainable
        weight = register_parameter("weight", torch::ones({1, 1, hidden_size}));
        // Optional: initialize weight to 1.0 (common practice)
        reset_parameters();
    }

    void reset_parameters() {
        torch::nn::init::ones_(weight);
    }

    torch::Tensor forward(torch::Tensor x) {
        // x is expected to be (B, T, H)
        // Our custom function handles everything (including broadcasting weight correctly)
        return rmsnorm(x, weight, eps)[0];   // rmsnorm returns a tensor_list with one element
    }

    // Expose eps if you want to change it later (optional)
    double eps;
    torch::Tensor weight;
};
*/

class FusedScanFunction : public torch::autograd::Function<FusedScanFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor log_coeffs,
        torch::Tensor log_values
    ) {
        TORCH_CHECK(log_coeffs.is_cuda(), "log_coeffs must be on CUDA");
        TORCH_CHECK(log_values.is_cuda(), "log_values must be on CUDA");
        TORCH_CHECK(log_coeffs.dtype() == log_values.dtype(), "dtypes must match");
        TORCH_CHECK(log_coeffs.dim() == 3 && log_values.dim() == 3, "must be (B, T, H)");
        TORCH_CHECK(log_values.size(1) == log_coeffs.size(1), "T must match");

        auto dtype = log_coeffs.dtype();        // e.g., kBFloat16 or kFloat32
        auto device = log_coeffs.device();      // e.g., cuda:0
        auto B = log_coeffs.size(0);
        auto T = log_coeffs.size(1);            // T = T_total (e.g. T+1)
        auto H = log_coeffs.size(2);

        // Output: same dtype and device as input
        auto out = torch::empty({B, T, H}, log_coeffs.options());

        // Intermediates: must be float32, but on same device!
        auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        //auto options_double = torch::TensorOptions().dtype(torch::kFloat64).device(device);
        auto a_star = torch::empty({B, T, H}, options_float);
        auto s_vals = torch::empty({B, T, H}, options_float);

        // Launch kernel
        if (dtype == torch::kFloat32) {
            launch_fused_scan_forward<float>(
                out.data_ptr<float>(),
                a_star.data_ptr<float>(),
                s_vals.data_ptr<float>(),
                log_coeffs.data_ptr<float>(),
                log_values.data_ptr<float>(),
                static_cast<int>(T),
                static_cast<int>(H),
                static_cast<int>(B)
            );

        } else if (dtype == torch::kBFloat16) {
            launch_fused_scan_forward<at::BFloat16>(
                out.data_ptr<at::BFloat16>(),
                a_star.data_ptr<float>(),
                s_vals.data_ptr<float>(),
                log_coeffs.data_ptr<at::BFloat16>(),
                log_values.data_ptr<at::BFloat16>(),
                static_cast<int>(T),
                static_cast<int>(H),
                static_cast<int>(B)
            );
        } else {
            TORCH_CHECK(false, "Unsupported dtype. Only float32 and bfloat16 supported.");
        }

        // Save for backward (a_star and s_vals are float32, but that's fine)
        ctx->save_for_backward({log_coeffs, log_values, out, a_star, s_vals});
            out = torch::nan_to_num(out, 0.0f, 0.0f, 0.0f);

        return {out};
    }
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto log_coeffs = saved[0].contiguous();
        auto log_values = saved[1].contiguous();
        auto out = saved[2].contiguous();
        auto a_star_buf = saved[3].contiguous();  // float tensor
        auto s_vals = saved[4].contiguous();      // float tensor

        auto grad_out = grad_outputs[0].contiguous();
        auto dtype = log_coeffs.dtype();

        auto B = log_coeffs.size(0);
        auto T = log_coeffs.size(1);
        auto H = log_coeffs.size(2);

        auto grad_log_coeffs = torch::empty_like(log_coeffs);
        auto grad_log_values = torch::empty_like(log_values);

        if (dtype == torch::kFloat32) {
            launch_fused_scan_backward<float>(
                grad_log_coeffs.data_ptr<float>(),
                grad_log_values.data_ptr<float>(),
                grad_out.data_ptr<float>(),
                log_coeffs.data_ptr<float>(),
                log_values.data_ptr<float>(),
                out.data_ptr<float>(),
                a_star_buf.data_ptr<float>(),
                s_vals.data_ptr<float>(),
                static_cast<int>(T), // Probably not needed
                static_cast<int>(H),
                static_cast<int>(B)
            );
        } else if (dtype == torch::kBFloat16) {
            launch_fused_scan_backward<at::BFloat16>(
                grad_log_coeffs.data_ptr<at::BFloat16>(),
                grad_log_values.data_ptr<at::BFloat16>(),
                grad_out.data_ptr<at::BFloat16>(),
                log_coeffs.data_ptr<at::BFloat16>(),
                log_values.data_ptr<at::BFloat16>(),
                out.data_ptr<at::BFloat16>(),
                a_star_buf.data_ptr<float>(),
                s_vals.data_ptr<float>(),
                T, H, B
            );
        } else {
            TORCH_CHECK(false, "Unsupported dtype");
        }

        return {grad_log_coeffs, grad_log_values};
    }
};

// Named entrypoint: fused_scan(log_coeffs, log_values) -> out
torch::autograd::tensor_list fused_scan(
    torch::Tensor log_coeffs,
    torch::Tensor log_values
) {
    return FusedScanFunction::apply(log_coeffs, log_values);
}

class LogCumsumExpFunction : public torch::autograd::Function<LogCumsumExpFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor x  // (B, T, H)
    ) {
        TORCH_CHECK(x.is_cuda(), "x must be on CUDA");
        auto dtype = x.dtype();
        auto device = x.device();
        auto B = x.size(0), T = x.size(1), H = x.size(2);

        auto out = torch::empty({B, T, H}, x.options());
        //auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        auto options_double = torch::TensorOptions().dtype(torch::kFloat64).device(device);
        auto s_buf = torch::empty({B, T, H}, options_double);

        if (dtype == torch::kFloat32) {
            launch_logcumsumexp_forward<float>(
                out.data_ptr<float>(),
                s_buf.data_ptr<double>(),
                x.data_ptr<float>(),
                (int)T, (int)H, (int)B
            );
        } else if (dtype == torch::kBFloat16) {
            launch_logcumsumexp_forward<at::BFloat16>(
                out.data_ptr<at::BFloat16>(),
                s_buf.data_ptr<double>(),
                x.data_ptr<at::BFloat16>(),
                (int)T, (int)H, (int)B
            );
        } else {
            TORCH_CHECK(false, "Only float32 and bfloat16 supported");
        }

        ctx->save_for_backward({x, out, s_buf});
        return {out};
    }
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto x = saved[0].contiguous();
        auto s_buf = saved[2].contiguous();  // s_buf is saved, out is not needed

        auto grad_out = grad_outputs[0].contiguous();
        auto dtype = x.dtype();
        auto B = x.size(0), T = x.size(1), H = x.size(2);

        auto grad_x = torch::empty_like(x);

        if (dtype == torch::kFloat32) {
            launch_logcumsumexp_backward<float>(
                grad_x.data_ptr<float>(),
                grad_out.data_ptr<float>(),
                x.data_ptr<float>(),
                s_buf.data_ptr<double>(),
                (int)T, (int)H, (int)B
            );
        } else if (dtype == torch::kBFloat16) {
            launch_logcumsumexp_backward<at::BFloat16>(
                grad_x.data_ptr<at::BFloat16>(),
                grad_out.data_ptr<at::BFloat16>(),
                x.data_ptr<at::BFloat16>(),
                s_buf.data_ptr<double>(),
                (int)T, (int)H, (int)B
            );
        } else {
            TORCH_CHECK(false, "Unsupported dtype in backward");
        }

        return {grad_x};
    }
};

// Entry point
torch::Tensor logcumsumexp_cuda(torch::Tensor x) {
    return LogCumsumExpFunction::apply(x)[0];
}

class PPOFusedLossFunction : public torch::autograd::Function<PPOFusedLossFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor logits,           // (N, T, A)
        torch::Tensor values_pred,      // (N, T, 1) or (N, T)
        torch::Tensor actions,          // (N, T)
        torch::Tensor old_logprobs,     // (N, T)
        torch::Tensor advantages,       // (N, T)
        torch::Tensor prio,             // (N, 1) — importance weights
        torch::Tensor values,           // (N, T)
        torch::Tensor returns,          // (N, T)
        torch::Tensor adv_mean,         // (1)
        torch::Tensor adv_std,          // (1)
        double clip_coef,
        double vf_clip_coef,
        double vf_coef,
        double ent_coef
    ) {
        TORCH_CHECK(logits.is_cuda(), "logits must be on CUDA");
        auto dtype = logits.dtype();
        TORCH_CHECK(dtype == torch::kFloat32 || dtype == torch::kBFloat16,
                    "Only float32 and bfloat16 supported");

        auto device = logits.device();
        auto N = logits.size(0);
        auto T = logits.size(1);
        auto A = logits.size(2);

        // Extract scalar hyperparams
        /*
        float adv_mean_val = adv_mean.item<float>();
        float adv_std_val = adv_std.item<float>();
        float clip_coef_val = clip_coef.item<float>();
        float vf_clip_coef_val = vf_clip_coef.item<float>();
        float vf_coef_val = vf_coef.item<float>();
        float ent_coef_val = ent_coef.item<float>();
        */

        // DO NOT let the compiler know these values at compile time
        /*
        float adv_mean = rand() / static_cast<float>(RAND_MAX);
        float adv_std = 0.5f + (rand() / static_cast<float>(RAND_MAX)) * 2.0f;  // [0.5, 2.5]
        float clip_coef = (rand() / static_cast<float>(RAND_MAX)) * 0.4f;       // [0.0, 0.4]
        float vf_clip_coef = (rand() / static_cast<float>(RAND_MAX)) * 0.4f;
        float vf_coef = (rand() / static_cast<float>(RAND_MAX)) * 1.0f;
        float ent_coef = (rand() / static_cast<float>(RAND_MAX)) * 0.1f;
        float adv_mean= 1.0f;
        float adv_std= 1.0f;
        float clip_coef= 1.0f;
        float vf_clip_coef= 1.0f;
        float vf_coef= 1.0f;
        float ent_coef= 1.0f;
        */

        // Output: scalar loss
        auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        auto options_double = torch::TensorOptions().dtype(torch::kFloat64).device(device);
        auto loss_output = torch::zeros({1}, options_float);

        // Saved for backward: (N, T, 5) → but use (N*T, 5) for flat indexing
        auto saved_for_backward = torch::empty({N * T, 5}, options_double);

        if (dtype == torch::kFloat32) {
            launch_ppo_loss_forward<float>(
                loss_output.data_ptr<float>(),
                saved_for_backward.data_ptr<double>(),
                logits.data_ptr<float>(),
                values_pred.data_ptr<float>(),
                actions.data_ptr<int64_t>(),
                old_logprobs.data_ptr<float>(),
                advantages.data_ptr<float>(),
                prio.data_ptr<float>(),      // (N, 1) → index as [n]
                values.data_ptr<float>(),
                returns.data_ptr<float>(),
                adv_mean.data_ptr<float>(),
                adv_std.data_ptr<float>(),
                clip_coef,
                vf_clip_coef,
                vf_coef,
                ent_coef,
                T, A, N
            );
        } else if (dtype == torch::kBFloat16) {
            launch_ppo_loss_forward<at::BFloat16>(
                loss_output.data_ptr<float>(),
                saved_for_backward.data_ptr<double>(),
                logits.data_ptr<at::BFloat16>(),
                values_pred.data_ptr<at::BFloat16>(),
                actions.data_ptr<int64_t>(),
                old_logprobs.data_ptr<at::BFloat16>(),
                advantages.data_ptr<at::BFloat16>(),
                prio.data_ptr<at::BFloat16>(),
                values.data_ptr<at::BFloat16>(),
                returns.data_ptr<at::BFloat16>(),
                adv_mean.data_ptr<float>(), // TODO: is this correct?
                adv_std.data_ptr<float>(),
                clip_coef,
                vf_clip_coef,
                vf_coef,
                ent_coef,
                T, A, N
            );
        }

        // Check errors
        /*
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA sync error: %s\n", cudaGetErrorString(err));
        }
        */

        // Compute mean loss: divide by (N * T)
        //float accumulated;
        //cudaMemcpy(&accumulated, loss_output.data_ptr<float>(), sizeof(float), cudaMemcpyDeviceToHost);
        //auto mean_loss = torch::tensor(accumulated / (N * T), options_float);

        // Save scalars
        ctx->saved_data["clip_coef"] = clip_coef;
        ctx->saved_data["vf_clip_coef"] = vf_clip_coef;
        ctx->saved_data["vf_coef"] = vf_coef;
        ctx->saved_data["ent_coef"] = ent_coef;

        // Save inputs and intermediates
        ctx->save_for_backward({logits, values_pred, actions, old_logprobs, advantages,
                                prio, values, returns, adv_mean, adv_std, saved_for_backward});

        return {loss_output / (N * T)};
    }
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto logits = saved[0].contiguous();           // (N, T, A)
        auto values_pred = saved[1].contiguous();      // (N, T, 1) or (N, T)
        auto actions = saved[2].contiguous();          // (N, T)
        auto old_logprobs = saved[3].contiguous();     // (N, T)
        auto advantages = saved[4].contiguous();       // (N, T)
        auto prio = saved[5].contiguous();             // (N, 1)
        auto values = saved[6].contiguous();           // (N, T)
        auto returns = saved[7].contiguous();          // (N, T)
        auto adv_mean = saved[8].contiguous();         // (1)
        auto adv_std = saved[9].contiguous();          // (1)
        auto saved_for_backward = saved[10].contiguous();  // (N*T, 5)

        auto dtype = logits.dtype();
        auto N = logits.size(0);
        auto T = logits.size(1);
        auto A = logits.size(2);

        float clip_coef = ctx->saved_data["clip_coef"].to<double>();
        float vf_clip_coef = ctx->saved_data["vf_clip_coef"].to<double>();
        float vf_coef = ctx->saved_data["vf_coef"].to<double>();
        float ent_coef = ctx->saved_data["ent_coef"].to<double>();

        auto grad_loss = grad_outputs[0].sum().to(torch::kFloat32).reshape({1});
        //auto grad_out_scalar = grad_outputs[0].sum();  // dL/d(loss)
        //auto grad_loss = torch::empty({1}, logits.options()).to(torch::kFloat32);
        //grad_loss.fill_(grad_out_scalar.item<float>());

        auto grad_logits = torch::empty_like(logits);
        auto grad_values_pred = torch::empty_like(values_pred);

        // TODO: Why are we passing grad loss in float?
        if (dtype == torch::kFloat32) {
            launch_ppo_loss_backward<float>(
                grad_logits.data_ptr<float>(),
                grad_values_pred.data_ptr<float>(),
                grad_loss.data_ptr<float>(),
                logits.data_ptr<float>(),
                actions.data_ptr<int64_t>(),
                old_logprobs.data_ptr<float>(),
                advantages.data_ptr<float>(),
                prio.data_ptr<float>(),
                values.data_ptr<float>(),
                returns.data_ptr<float>(),
                saved_for_backward.data_ptr<double>(),
                adv_mean.data_ptr<float>(),
                adv_std.data_ptr<float>(),
                clip_coef, vf_clip_coef,
                vf_coef, ent_coef,
                T, A, N
            );
        } else if (dtype == torch::kBFloat16) {
            launch_ppo_loss_backward<at::BFloat16>(
                grad_logits.data_ptr<at::BFloat16>(),
                grad_values_pred.data_ptr<at::BFloat16>(),
                grad_loss.data_ptr<float>(),
                logits.data_ptr<at::BFloat16>(),
                actions.data_ptr<int64_t>(),
                old_logprobs.data_ptr<at::BFloat16>(),
                advantages.data_ptr<at::BFloat16>(),
                prio.data_ptr<at::BFloat16>(),
                values.data_ptr<at::BFloat16>(),
                returns.data_ptr<at::BFloat16>(),
                saved_for_backward.data_ptr<double>(),
                adv_mean.data_ptr<float>(),
                adv_std.data_ptr<float>(),
                clip_coef, vf_clip_coef,
                vf_coef, ent_coef,
                T, A, N
            );
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Backward kernel error: %s\n", cudaGetErrorString(err));
        }
        /*
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "Backward sync error: %s\n", cudaGetErrorString(err));
        }
        */

        return {
            grad_logits,
            grad_values_pred,
            {}, {}, {}, {}, {}, {},  // actions, old_logprobs, advantages, prio, values, returns
            {}, {}, {}, {}, {}, {}   // adv_mean, adv_std, clip_coef, vf_clip_coef, vf_coef, ent_coef
        };
    }
};

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
) {
    return PPOFusedLossFunction::apply(logits, values_pred, actions,
        old_logprobs, advantages, prio, values, returns, adv_mean,
        adv_std, clip_coef, vf_clip_coef, vf_coef, ent_coef);
}
