// profile_torch_lib.cu
// Static torch library for profiler — compiled once (slow), linked by profile_kernels.cu (fast).
// Provides correctness checks (kernel vs torch cpp reference) and cpp benchmark timing.
//
// Build:
//   python setup.py build_profile_torch

#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "pufferlib/src/models.cu"
#include "pufferlib/src/legacy/legacy_modules.h"

using namespace pufferlib;

#include "profile_common.h"

static const bool USE_BF16 = (PRECISION_DTYPE == torch::kBFloat16);

// Helper: profile_kernel + empty torch cache
static float profile_kernel_torch(kernel_fn fn, void* args, const char* name = nullptr) {
    float ms = profile_kernel(fn, args, name);
    c10::cuda::CUDACachingAllocator::emptyCache();
    return ms;
}

// ============================================================================
// MinGRU gate
// ============================================================================

struct MingruTorch {
    Tensor state, combined;
    int B, H;
};

static void run_mingru_cpp(MingruTorch* a) {
    torch::NoGradGuard g;
    mingru_gate_cpp(a->state, a->combined);
}

extern "C" void torch_bench_mingrugate(MingruGateArgs* raw) {
    auto opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);
    MingruTorch t;
    t.B = raw->B; t.H = raw->H;
    t.state = torch::from_blob(raw->state, {raw->B, raw->H}, opts);
    t.combined = torch::from_blob(raw->combined, {raw->B, 3*raw->H}, opts);

    // Correctness: run kernel, run cpp, compare
    // Kernel writes to raw->out / raw->next_state
    run_mingrugate_forward(raw);
    auto k_out = torch::from_blob(raw->out, {raw->B, raw->H}, opts).clone();
    auto k_ns  = torch::from_blob(raw->next_state, {raw->B, raw->H}, opts).clone();

    auto cpp_out = mingru_gate_cpp(t.state.to(torch::kFloat32), t.combined.to(torch::kFloat32));
    float rtol = USE_BF16 ? 5e-2f : 1e-3f, atol = USE_BF16 ? 1e-2f : 1e-4f;
    bool ok_out = torch::allclose(k_out.to(torch::kFloat32), cpp_out[0], rtol, atol);
    float d_out = (k_out.to(torch::kFloat32) - cpp_out[0]).abs().max().item<float>();
    bool ok_ns = torch::allclose(k_ns.to(torch::kFloat32), cpp_out[1], rtol, atol);
    float d_ns = (k_ns.to(torch::kFloat32) - cpp_out[1]).abs().max().item<float>();
    printf("  correctness: out=%s(%.2e) next_state=%s(%.2e)\n",
           ok_out ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", d_out,
           ok_ns  ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", d_ns);

    float cpp_ms = profile_kernel_torch((kernel_fn)run_mingru_cpp, &t);
    print_timing("forward (cpp)", cpp_ms, raw->B);
}

// ============================================================================
// LogCumsumExp
// ============================================================================

struct LogcumsumexpTorch {
    Tensor x, grad_out;
    int B, T, H;
};

static void run_logcumsumexp_cpp_fn(LogcumsumexpTorch* a) {
    torch::NoGradGuard g;
    logcumsumexp_cpp(a->x);
}

extern "C" void torch_bench_logcumsumexp(LogcumsumexpArgs* raw) {
    auto opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);
    LogcumsumexpTorch t;
    t.B = raw->B; t.T = raw->T; t.H = raw->H;
    t.x = torch::from_blob(raw->x, {raw->B, raw->T, raw->H}, opts).clone();
    t.grad_out = torch::from_blob(raw->grad_out, {raw->B, raw->T, raw->H}, opts).clone();
    int batch_seq = raw->B * raw->T;

    // Correctness: forward
    run_logcumsumexp_forward(raw);
    auto k_out = torch::from_blob(raw->out, {raw->B, raw->T, raw->H}, opts).clone();
    auto cpp_out = logcumsumexp_cpp(t.x.to(torch::kFloat32));

    float rtol = USE_BF16 ? 5e-2f : 1e-3f, atol = USE_BF16 ? 1e-2f : 1e-4f;
    float d = (k_out.to(torch::kFloat32) - cpp_out).abs().max().item<float>();
    bool ok = torch::allclose(k_out.to(torch::kFloat32), cpp_out, rtol, atol);
    printf("  forward correctness: %s(%.2e)\n",
           ok ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", d);

    // Correctness: backward
    auto x_k = t.x.clone().requires_grad_(true);
    auto out_k_ag = logcumsumexp_cpp(x_k);  // use cpp for autograd (kernel has no autograd)
    out_k_ag.backward(t.grad_out.to(x_k.dtype()));

    auto x_c = t.x.to(torch::kFloat32).clone().requires_grad_(true);
    auto out_c = logcumsumexp_cpp(x_c);
    out_c.backward(t.grad_out.to(torch::kFloat32));

    float gd = (x_k.grad().to(torch::kFloat32) - x_c.grad()).abs().max().item<float>();
    bool gok = torch::allclose(x_k.grad().to(torch::kFloat32), x_c.grad(), rtol, atol);
    printf("  backward correctness: %s(%.2e)\n",
           gok ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", gd);

    float cpp_ms = profile_kernel_torch((kernel_fn)run_logcumsumexp_cpp_fn, &t);
    print_timing("forward (cpp)", cpp_ms, batch_seq);
}

// ============================================================================
// Fused scan
// ============================================================================

struct FusedScanTorch {
    Tensor combined, state, grad_out, grad_next_state;
    int B, T, H;
};

static void run_fusedscan_cpp_fn(FusedScanTorch* a) {
    torch::NoGradGuard g;
    fused_scan_cpp(a->combined, a->state);
}

extern "C" void torch_bench_fusedscan(FusedScanArgs* raw) {
    auto opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);
    FusedScanTorch t;
    t.B = raw->B; t.T = raw->T; t.H = raw->H;
    t.combined = torch::from_blob(raw->combined, {raw->B, raw->T, 3*raw->H}, opts).clone();
    t.state = torch::from_blob(raw->state, {raw->B, 1, raw->H}, opts).clone();
    t.grad_out = torch::from_blob(raw->grad_out, {raw->B, raw->T, raw->H}, opts).clone();
    t.grad_next_state = torch::from_blob(raw->grad_next_state, {raw->B, 1, raw->H}, opts).clone();
    int batch_seq = raw->B * raw->T;

    // Correctness: forward
    run_fusedscan_forward(raw);
    auto k_out = torch::from_blob(raw->out, {raw->B, raw->T, raw->H}, opts).clone();
    auto k_ns  = torch::from_blob(raw->next_state, {raw->B, 1, raw->H}, opts).clone();

    auto c = fused_scan_cpp(t.combined.to(torch::kFloat32), t.state.to(torch::kFloat32));

    float rtol = USE_BF16 ? 1e-1f : 1e-3f, atol = USE_BF16 ? 5e-2f : 2e-4f;
    float d_out = (k_out.to(torch::kFloat32) - c[0]).abs().max().item<float>();
    bool ok_out = torch::allclose(k_out.to(torch::kFloat32), c[0], rtol, atol);
    float d_ns = (k_ns.to(torch::kFloat32) - c[1]).abs().max().item<float>();
    bool ok_ns = torch::allclose(k_ns.to(torch::kFloat32), c[1], rtol, atol);
    printf("  forward correctness: out=%s(%.2e) next_state=%s(%.2e)\n",
           ok_out ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", d_out,
           ok_ns  ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", d_ns);

    // Correctness: backward (via cpp autograd)
    auto comb_k = t.combined.clone().requires_grad_(true);
    auto st_k   = t.state.clone().requires_grad_(true);
    auto fwd_k  = fused_scan_cpp(comb_k, st_k);
    torch::autograd::backward({fwd_k[0], fwd_k[1]},
        {t.grad_out.to(comb_k.dtype()), t.grad_next_state.to(st_k.dtype())});

    auto comb_c = t.combined.to(torch::kFloat32).clone().requires_grad_(true);
    auto st_c   = t.state.to(torch::kFloat32).clone().requires_grad_(true);
    auto fwd_c  = fused_scan_cpp(comb_c, st_c);
    torch::autograd::backward({fwd_c[0], fwd_c[1]},
        {t.grad_out.to(torch::kFloat32), t.grad_next_state.to(torch::kFloat32)});

    float gc_d = (comb_k.grad().to(torch::kFloat32) - comb_c.grad()).abs().max().item<float>();
    bool gc_ok = torch::allclose(comb_k.grad().to(torch::kFloat32), comb_c.grad(), rtol, atol);
    float gs_d = (st_k.grad().to(torch::kFloat32) - st_c.grad()).abs().max().item<float>();
    bool gs_ok = torch::allclose(st_k.grad().to(torch::kFloat32), st_c.grad(), rtol, atol);
    printf("  backward correctness: grad_combined=%s(%.2e) grad_state=%s(%.2e)\n",
           gc_ok ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", gc_d,
           gs_ok ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", gs_d);

    float cpp_ms = profile_kernel_torch((kernel_fn)run_fusedscan_cpp_fn, &t);
    print_timing("forward (cpp)", cpp_ms, batch_seq);
}

// ============================================================================
// FCMax
// ============================================================================

struct FCMaxTorch {
    Tensor x, W, b, grad_out;
    int B, N, D_in, D_out;
};

static void run_fcmax_cpp_fn(FCMaxTorch* a) {
    torch::NoGradGuard g;
    fc_max_cpp(a->x.to(torch::kFloat32), a->W, a->b);
}

extern "C" void torch_bench_fcmax(FCMaxArgs* raw) {
    // TODO: cuBLAS crashes with these dimensions — investigate separately
    printf("  (skipped — cuBLAS issue under investigation)\n");
}

// ============================================================================
// PPO loss
// ============================================================================

struct PPOLossTorch {
    Tensor logits, values_pred, actions, old_logprobs, advantages;
    Tensor prio, values, returns;
    Tensor ratio_out, newvalue_out, act_sizes, losses;
    float clip_coef, vf_clip_coef, vf_coef, ent_coef;
    int N, T, A;
};

static void run_ppoloss_cpp_fn(PPOLossTorch* a) {
    torch::NoGradGuard g;
    fused_ppo_loss_cpp(
        a->logits, torch::Tensor(), a->values_pred, a->actions,
        a->old_logprobs, a->advantages, a->prio,
        a->values, a->returns,
        a->ratio_out, a->newvalue_out, a->act_sizes, a->losses,
        a->clip_coef, a->vf_clip_coef, a->vf_coef, a->ent_coef);
}

extern "C" void torch_bench_ppoloss(PPOLossArgs* raw) {
    auto opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);
    int NT = raw->N * raw->T;
    PPOLossTorch t;
    t.N = raw->N; t.T = raw->T; t.A = raw->A;
    t.clip_coef = raw->clip_coef; t.vf_clip_coef = raw->vf_clip_coef;
    t.vf_coef = raw->vf_coef; t.ent_coef = raw->ent_coef;

    t.logits = torch::from_blob(raw->logits, {raw->N, raw->T, raw->A}, opts).clone();
    t.values_pred = torch::from_blob(raw->values_pred, {raw->N, raw->T, 1}, opts).clone();
    t.actions = torch::from_blob(raw->actions, {raw->N, raw->T, 1}, cuda_f64).clone();
    t.old_logprobs = torch::from_blob(raw->old_logprobs, {raw->N, raw->T}, opts).clone();
    t.advantages = torch::from_blob(raw->advantages, {raw->N, raw->T}, cuda_f32).clone();
    t.prio = torch::from_blob(raw->prio, {raw->N, 1}, opts).clone();
    t.values = torch::from_blob(raw->values, {raw->N, raw->T}, opts).clone();
    t.returns = torch::from_blob(raw->returns, {raw->N, raw->T}, opts).clone();
    t.ratio_out = torch::zeros({raw->N, raw->T}, opts);
    t.newvalue_out = torch::zeros({raw->N, raw->T}, opts);
    t.act_sizes = torch::tensor({raw->A}, cuda_i32);
    t.losses = torch::zeros({LOSS_N + 1}, cuda_f32);

    // Correctness: compare cpp at native precision vs cpp at float32
    auto ratio_n = t.ratio_out.clone(), nv_n = t.newvalue_out.clone(), losses_n = t.losses.clone();
    auto loss_native = fused_ppo_loss_cpp(
        t.logits, torch::Tensor(), t.values_pred, t.actions,
        t.old_logprobs, t.advantages, t.prio,
        t.values, t.returns, ratio_n, nv_n, t.act_sizes, losses_n,
        t.clip_coef, t.vf_clip_coef, t.vf_coef, t.ent_coef)[0];

    auto ratio_f = torch::zeros({raw->N, raw->T}, cuda_f32);
    auto nv_f = torch::zeros({raw->N, raw->T}, cuda_f32);
    auto losses_f = torch::zeros({LOSS_N + 1}, cuda_f32);
    auto loss_f32 = fused_ppo_loss_cpp(
        t.logits.to(torch::kFloat32), torch::Tensor(), t.values_pred.to(torch::kFloat32),
        t.actions, t.old_logprobs.to(torch::kFloat32), t.advantages,
        t.prio.to(torch::kFloat32), t.values.to(torch::kFloat32),
        t.returns.to(torch::kFloat32), ratio_f, nv_f,
        t.act_sizes, losses_f,
        t.clip_coef, t.vf_clip_coef, t.vf_coef, t.ent_coef)[0];

    float d = (loss_native.to(torch::kFloat32) - loss_f32).abs().item<float>();
    bool ok = d < 1e-2f;
    printf("  forward correctness: loss=%s(%.2e)\n",
           ok ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", d);

    float cpp_ms = profile_kernel_torch((kernel_fn)run_ppoloss_cpp_fn, &t);
    print_timing("forward (cpp)", cpp_ms, NT);
}

// ============================================================================
// Sample logits
// ============================================================================

struct SampleLogitsTorch {
    Tensor logits, act_sizes_cpu;
    int B, A;
};

static void run_samplelogits_cpp_fn(SampleLogitsTorch* a) {
    torch::NoGradGuard g;
    sample_discrete_cpp(a->logits, a->act_sizes_cpu, 1);
}

extern "C" void torch_bench_samplelogits(SampleLogitsArgs* raw) {
    auto opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);
    SampleLogitsTorch t;
    t.B = raw->B; t.A = raw->A;
    t.logits = torch::from_blob(raw->logits, {raw->B, raw->A}, opts);
    t.act_sizes_cpu = torch::tensor({(int64_t)raw->A}, torch::dtype(torch::kInt64));

    // Correctness: run kernel, verify logprobs match log_softmax
    run_samplelogits_forward(raw);
    auto actions = torch::from_blob(raw->actions, {raw->B, 1}, cuda_f64);
    auto logprobs = torch::from_blob(raw->logprobs, {raw->B}, opts);
    auto value_out = torch::from_blob(raw->value_out, {raw->B}, opts);
    auto value_in  = torch::from_blob(raw->value, {raw->B, 1}, opts);

    auto log_probs = torch::log_softmax(t.logits.to(torch::kFloat32), 1);
    auto expected = log_probs.gather(1, actions.to(torch::kInt64)).squeeze(1);
    auto actual = logprobs.to(torch::kFloat32);

    float rtol = 1e-2f, atol = 1e-3f;
    float d = (actual - expected).abs().max().item<float>();
    bool ok = torch::allclose(actual, expected, rtol, atol);
    printf("  logprob consistency: %s(%.2e)\n",
           ok ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", d);

    float vd = (value_out.to(torch::kFloat32) - value_in.squeeze(1).to(torch::kFloat32)).abs().max().item<float>();
    bool vok = torch::allclose(value_out.to(torch::kFloat32), value_in.squeeze(1).to(torch::kFloat32), rtol, atol);
    printf("  value passthrough: %s(%.2e)\n",
           vok ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", vd);

    auto actions_flat = actions.to(torch::kInt64).flatten();
    bool valid = (actions_flat.ge(0) & actions_flat.lt(raw->A)).all().item<bool>();
    printf("  action validity: %s\n", valid ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m");

    float cpp_ms = profile_kernel_torch((kernel_fn)run_samplelogits_cpp_fn, &t);
    print_timing("forward (cpp)", cpp_ms, raw->B);
}

// ============================================================================
// Composite profiles — stubs (no torch Policy on this branch)
// ============================================================================

extern "C" void torch_profile_forwardcall(int batch, int input_size, int hidden, int act_n, int num_layers) {
    printf("forward_call: not available (no torch Policy on static-native branch)\n\n");
}

extern "C" void torch_profile_trainforward(int N, int T, int input, int hidden, int act_n, int layers) {
    printf("trainforward: not available (no torch Policy on static-native branch)\n\n");
}

extern "C" void torch_profile_trainstep(int N, int T, int input, int hidden, int act_n, int layers) {
    printf("trainstep: not available (no torch Policy on static-native branch)\n\n");
}

extern "C" void torch_profile_rolloutcopy(int S, int T, int mb_segs, int input, int A, int layers, int H) {
    printf("rolloutcopy: not available (no torch Policy on static-native branch)\n\n");
}
