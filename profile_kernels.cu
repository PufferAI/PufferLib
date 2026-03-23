// profile_kernels.cu
// Fast-compile kernel profiling binary. Links against libprofile_torch.a for
// torch-dependent correctness checks and composite profiles.
//
// Build without torch (fast):
//   nvcc -O3 -arch=sm_89 -DPRECISION_FLOAT -I. profile_kernels.cu -o profile
//
// Build with torch (links static lib):
//   python setup.py build_profiler
//
// Usage:
//   ./profile <profile>
//   ./profile kernels          # All individual kernel profiles
//   ./profile trainforward     # Training forward+backward breakdown
//   ./profile trainstep        # Full training step with Muon optimizer
//   ./profile rolloutcopy      # Per-minibatch data prep: advantage+prio+copy
//   ./profile forwardcall      # Inference forward pass
//   ./profile envspeed         # Environment throughput
//   ./profile all              # Everything

// ============================================================================
// Section 1: Shared infrastructure + kernel includes
// ============================================================================

#include "profile_common.h"
#include "pufferlib/src/kernels.cu"

// ============================================================================
// Section 2: MinGRU gate — raw kernel create/free/run
// ============================================================================

MingruGateArgs* create_mingrugateargs(int batch, int hidden) {
    MingruGateArgs* args = (MingruGateArgs*)calloc(1, sizeof(MingruGateArgs));
    args->B = batch;
    args->H = hidden;

    int N_state = batch * hidden;
    int N_combined = batch * 3 * hidden;

    cudaMalloc(&args->state, N_state * sizeof(precision_t));
    cudaMalloc(&args->combined, N_combined * sizeof(precision_t));
    cudaMalloc(&args->out, N_state * sizeof(precision_t));
    cudaMalloc(&args->next_state, N_state * sizeof(precision_t));

    float* state_buf = (float*)malloc(N_state * sizeof(float));
    float* combined_buf = (float*)malloc(N_combined * sizeof(float));

    for (int i = 0; i < N_state; ++i) {
        state_buf[i] = fabsf(rand1()) + 0.1f;
    }
    for (int b = 0; b < batch; ++b) {
        int base = b * 3 * hidden;
        for (int h = 0; h < hidden; ++h) {
            combined_buf[base + h] = rand1() * 5.0f;
            combined_buf[base + hidden + h] = rand1() * 5.0f;
            combined_buf[base + 2 * hidden + h] = rand1() * 2.0f;
        }
    }

    float_to_device(args->state, state_buf, N_state);
    float_to_device(args->combined, combined_buf, N_combined);

    free(state_buf);
    free(combined_buf);
    return args;
}

void free_mingrugateargs(MingruGateArgs* args) {
    cudaFree(args->state);
    cudaFree(args->combined);
    cudaFree(args->out);
    cudaFree(args->next_state);
    free(args);
}

void run_mingrugate_forward(MingruGateArgs* args) {
    mingru_gate_inference_kernel<<<grid_size(args->B * args->H), BLOCK_SIZE>>>(
        args->out, args->next_state, args->combined, args->state,
        args->H, args->B);
}

void profile_mingrugate(int batch, int hidden) {
    printf("mingru_gate (B=%d, H=%d, combined=%dx%d)\n", batch, hidden, batch, 3*hidden);

    MingruGateArgs* args = create_mingrugateargs(batch, hidden);
    float fwd_ms = profile_kernel((kernel_fn)run_mingrugate_forward, args);
    print_timing("forward", fwd_ms, batch);

#ifdef USE_TORCH
    torch_bench_mingrugate(args);
#endif
    printf("\n");
    free_mingrugateargs(args);
}

// ============================================================================
// Section 3: LogCumsumExp — raw kernel create/free/run
// ============================================================================

LogcumsumexpArgs* create_logcumsumexpargs(int batch, int seq, int hidden) {
    LogcumsumexpArgs* args = (LogcumsumexpArgs*)calloc(1, sizeof(LogcumsumexpArgs));
    args->B = batch;
    args->T = seq;
    args->H = hidden;
    args->N = batch * seq * hidden;

    cudaMalloc(&args->x, args->N * sizeof(precision_t));
    cudaMalloc(&args->out, args->N * sizeof(precision_t));
    cudaMalloc(&args->s_buf, args->N * sizeof(double));
    cudaMalloc(&args->grad_x, args->N * sizeof(precision_t));
    cudaMalloc(&args->grad_out, args->N * sizeof(precision_t));

    float* buf = (float*)malloc(args->N * sizeof(float) * 2);
    float* x_buf = buf;
    float* grad_out_buf = buf + args->N;
    for (int i = 0; i < args->N; ++i) {
        x_buf[i] = rand1();
        grad_out_buf[i] = rand1();
    }

    float_to_device(args->x, x_buf, args->N);
    float_to_device(args->grad_out, grad_out_buf, args->N);

    free(buf);
    return args;
}

void free_logcumsumexpargs(LogcumsumexpArgs* args) {
    cudaFree(args->x);
    cudaFree(args->out);
    cudaFree(args->s_buf);
    cudaFree(args->grad_x);
    cudaFree(args->grad_out);
    free(args);
}

void run_logcumsumexp_forward(LogcumsumexpArgs* args) {
    logcumsumexp_forward_kernel<<<grid_size(args->B * args->H), BLOCK_SIZE>>>(
        args->out, args->s_buf, args->x, args->T, args->H, args->B);
}

void run_logcumsumexp_backward(LogcumsumexpArgs* args) {
    logcumsumexp_backward_kernel<<<grid_size(args->B * args->H), BLOCK_SIZE>>>(
        args->grad_x, args->grad_out, args->x, args->s_buf, args->T, args->H, args->B);
}

void profile_logcumsumexp(int batch, int seq, int hidden) {
    LogcumsumexpArgs* args = create_logcumsumexpargs(batch, seq, hidden);
    printf("logcumsumexp (N=%d, %dx%dx%d)\n", args->N, batch, seq, hidden);

    float fwd_ms = profile_kernel((kernel_fn)run_logcumsumexp_forward, args);
    print_timing("forward", fwd_ms, batch*seq);

    float bwd_ms = profile_kernel((kernel_fn)run_logcumsumexp_backward, args);
    print_timing("backward", bwd_ms, batch*seq);

#ifdef USE_TORCH
    torch_bench_logcumsumexp(args);
#endif
    printf("\n");
    free_logcumsumexpargs(args);
}

// ============================================================================
// Section 4: Fused scan — raw kernel create/free/run
// ============================================================================

FusedScanArgs* create_fusedscanargs(int batch, int seq, int hidden) {
    FusedScanArgs* args = (FusedScanArgs*)calloc(1, sizeof(FusedScanArgs));
    args->B = batch;
    args->T = seq;
    args->H = hidden;
    args->N = batch * seq * hidden;

    int N_combined = batch * seq * 3 * hidden;
    int N_state = batch * hidden;
    int N_buf = batch * (seq + 1) * hidden;

    cudaMalloc(&args->combined, N_combined * sizeof(precision_t));
    cudaMalloc(&args->state, N_state * sizeof(precision_t));
    cudaMalloc(&args->out, args->N * sizeof(precision_t));
    cudaMalloc(&args->next_state, N_state * sizeof(precision_t));
    cudaMalloc(&args->a_star, N_buf * sizeof(float));
    cudaMalloc(&args->s_vals, N_buf * sizeof(float));
    cudaMalloc(&args->log_values_buf, N_buf * sizeof(float));
    cudaMalloc(&args->grad_combined, N_combined * sizeof(precision_t));
    cudaMalloc(&args->grad_state, N_state * sizeof(precision_t));
    cudaMalloc(&args->grad_out, args->N * sizeof(precision_t));
    cudaMalloc(&args->grad_next_state, N_state * sizeof(precision_t));

    float* combined_buf = (float*)malloc(N_combined * sizeof(float));
    float* state_buf = (float*)malloc(N_state * sizeof(float));
    float* grad_out_buf = (float*)malloc(args->N * sizeof(float));
    float* grad_next_state_buf = (float*)malloc(N_state * sizeof(float));

    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < seq; ++t) {
            for (int h = 0; h < hidden; ++h) {
                int base = b * seq * 3 * hidden + t * 3 * hidden;
                combined_buf[base + h] = rand1() * 5.0f;
                combined_buf[base + hidden + h] = rand1() * 5.0f;
                combined_buf[base + 2 * hidden + h] = rand1() * 2.0f;
            }
        }
    }
    for (int i = 0; i < N_state; ++i) state_buf[i] = fabsf(rand1()) + 0.1f;
    for (int i = 0; i < args->N; ++i) grad_out_buf[i] = rand1();
    for (int i = 0; i < N_state; ++i) grad_next_state_buf[i] = rand1();

    float_to_device(args->combined, combined_buf, N_combined);
    float_to_device(args->state, state_buf, N_state);
    float_to_device(args->grad_out, grad_out_buf, args->N);
    float_to_device(args->grad_next_state, grad_next_state_buf, N_state);

    free(combined_buf);
    free(state_buf);
    free(grad_out_buf);
    free(grad_next_state_buf);
    return args;
}

void free_fusedscanargs(FusedScanArgs* args) {
    cudaFree(args->combined);
    cudaFree(args->state);
    cudaFree(args->out);
    cudaFree(args->next_state);
    cudaFree(args->a_star);
    cudaFree(args->s_vals);
    cudaFree(args->log_values_buf);
    cudaFree(args->grad_combined);
    cudaFree(args->grad_state);
    cudaFree(args->grad_out);
    cudaFree(args->grad_next_state);
    free(args);
}

void run_fusedscan_forward(FusedScanArgs* args) {
    fused_scan_forward_kernel_checkpointed<<<grid_size(args->B * args->H), BLOCK_SIZE>>>(
        args->out, args->next_state,
        args->a_star, args->s_vals, args->log_values_buf,
        args->combined, args->state,
        args->T, args->H, args->B);
}

void run_fusedscan_backward(FusedScanArgs* args) {
    fused_scan_backward_kernel_checkpointed<<<grid_size(args->B * args->H), BLOCK_SIZE>>>(
        args->grad_combined, args->grad_state,
        args->grad_out, args->grad_next_state,
        args->combined, args->state,
        args->a_star, args->s_vals, args->log_values_buf,
        args->T, args->H, args->B);
}

void profile_fusedscan(int batch, int seq, int hidden) {
    FusedScanArgs* args = create_fusedscanargs(batch, seq, hidden);
    printf("fused_scan (N=%d, %dx%dx%d, combined=%dx%dx%d)\n",
           args->N, batch, seq, hidden, batch, seq, 3*hidden);

    float fwd_ms = profile_kernel((kernel_fn)run_fusedscan_forward, args);
    print_timing("forward", fwd_ms, batch*seq);

    float bwd_ms = profile_kernel((kernel_fn)run_fusedscan_backward, args);
    print_timing("backward", bwd_ms, batch*seq);

#ifdef USE_TORCH
    torch_bench_fusedscan(args);
#endif
    printf("\n");
    free_fusedscanargs(args);
}

// ============================================================================
// Section 5: FCMax — raw kernel create/free/run
// ============================================================================

FCMaxArgs* create_fcmaxargs(int batch, int num_points, int d_in, int d_out) {
    FCMaxArgs* args = (FCMaxArgs*)calloc(1, sizeof(FCMaxArgs));
    args->B = batch;
    args->N = num_points;
    args->D_in = d_in;
    args->D_out = d_out;

    int N_x = batch * num_points * d_in;
    int N_W = d_out * d_in;
    int N_out = batch * d_out;

    cudaMalloc(&args->x, N_x * sizeof(precision_t));
    cudaMalloc(&args->W, N_W * sizeof(float));
    cudaMalloc(&args->b, d_out * sizeof(float));
    cudaMalloc(&args->out, N_out * sizeof(precision_t));
    cudaMalloc(&args->argmax_indices, N_out * sizeof(int));
    cudaMalloc(&args->grad_x, N_x * sizeof(float));
    cudaMalloc(&args->grad_W, N_W * sizeof(float));
    cudaMalloc(&args->grad_b, d_out * sizeof(float));
    cudaMalloc(&args->grad_out, N_out * sizeof(precision_t));

    float* x_buf = (float*)malloc(N_x * sizeof(float));
    float* W_buf = (float*)malloc(N_W * sizeof(float));
    float* b_buf = (float*)malloc(d_out * sizeof(float));
    float* grad_out_buf = (float*)malloc(N_out * sizeof(float));

    for (int i = 0; i < N_x; ++i) x_buf[i] = rand1();
    for (int i = 0; i < N_W; ++i) W_buf[i] = rand1() * 0.1f;
    for (int i = 0; i < d_out; ++i) b_buf[i] = 0.0f;
    for (int i = 0; i < N_out; ++i) grad_out_buf[i] = rand1();

    float_to_device(args->x, x_buf, N_x);
    cudaMemcpy(args->W, W_buf, N_W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->b, b_buf, d_out * sizeof(float), cudaMemcpyHostToDevice);
    float_to_device(args->grad_out, grad_out_buf, N_out);

    free(x_buf);
    free(W_buf);
    free(b_buf);
    free(grad_out_buf);
    return args;
}

void free_fcmaxargs(FCMaxArgs* args) {
    cudaFree(args->x);
    cudaFree(args->W);
    cudaFree(args->b);
    cudaFree(args->out);
    cudaFree(args->argmax_indices);
    cudaFree(args->grad_x);
    cudaFree(args->grad_W);
    cudaFree(args->grad_b);
    cudaFree(args->grad_out);
    free(args);
}

void run_fcmax_forward(FCMaxArgs* args) {
    fc_max_forward_kernel<<<grid_size(args->B * args->D_out), BLOCK_SIZE>>>(
        args->out, args->argmax_indices,
        args->x, args->W, args->b,
        args->B, args->N, args->D_in, args->D_out);
}

void run_fcmax_backward(FCMaxArgs* args) {
    fc_max_backward_grad_b_kernel<<<grid_size(args->D_out), BLOCK_SIZE>>>(
        args->grad_b, args->grad_out, args->B, args->D_out);

    fc_max_backward_grad_W_kernel<<<grid_size(args->D_out * args->D_in), BLOCK_SIZE>>>(
        args->grad_W, args->grad_out, args->x, args->argmax_indices,
        args->B, args->N, args->D_in, args->D_out);

    fc_max_backward_grad_x_kernel<<<grid_size(args->B * args->D_in), BLOCK_SIZE>>>(
        args->grad_x, args->grad_out, args->W, args->argmax_indices,
        args->B, args->N, args->D_in, args->D_out);
}

void profile_fcmax(int batch, int num_points, int d_in, int d_out) {
    FCMaxArgs* args = create_fcmaxargs(batch, num_points, d_in, d_out);

    printf("fc_max (B=%d, N=%d, D_in=%d, D_out=%d)\n", batch, num_points, d_in, d_out);

    float fwd_ms = profile_kernel((kernel_fn)run_fcmax_forward, args);
    print_timing("forward", fwd_ms, batch);

    float bwd_ms = profile_kernel((kernel_fn)run_fcmax_backward, args);
    print_timing("backward", bwd_ms, batch);

#ifdef USE_TORCH
    torch_bench_fcmax(args);
#endif
    printf("\n");

    free_fcmaxargs(args);
}

// ============================================================================
// Section 6: PPO loss — raw kernel create/free/run
// ============================================================================

PPOLossArgs* create_ppolossargs(int batch, int seq, int actions) {
    PPOLossArgs* args = (PPOLossArgs*)calloc(1, sizeof(PPOLossArgs));
    args->N = batch;
    args->T = seq;
    args->A = actions;
    args->num_atns = 1;

    int NT = batch*seq;
    int NTA = batch*seq * actions;

    cudaMalloc(&args->logits, NTA * sizeof(precision_t));
    cudaMalloc(&args->values_pred, NT * sizeof(precision_t));
    cudaMalloc(&args->actions, NT * sizeof(double));
    cudaMalloc(&args->old_logprobs, NT * sizeof(precision_t));
    cudaMalloc(&args->advantages, NT * sizeof(float));
    cudaMalloc(&args->prio, batch * sizeof(precision_t));
    cudaMalloc(&args->values, NT * sizeof(precision_t));
    cudaMalloc(&args->returns, NT * sizeof(precision_t));
    cudaMalloc(&args->adv_mean, sizeof(float));
    cudaMalloc(&args->adv_var, sizeof(float));
    cudaMalloc(&args->loss, sizeof(float));
    cudaMalloc(&args->losses_acc, LOSS_N * sizeof(float));
    cudaMemset(args->losses_acc, 0, LOSS_N * sizeof(float));
    cudaMalloc(&args->saved_for_backward, NT * 5 * sizeof(double));
    cudaMalloc(&args->ratio_out, NT * sizeof(precision_t));
    cudaMalloc(&args->newvalue_out, NT * sizeof(precision_t));
    cudaMalloc(&args->grad_logits, NTA * sizeof(float));
    cudaMalloc(&args->grad_values_pred, NT * sizeof(float));
    cudaMalloc(&args->grad_loss, sizeof(float));
    cudaMalloc(&args->act_sizes, sizeof(int));

    cudaMemcpy(args->act_sizes, &actions, sizeof(int), cudaMemcpyHostToDevice);

    float* buf = (float*)malloc((NTA + NT * 5 + batch) * sizeof(float));
    float* logits_buf = buf;
    float* values_pred_buf = buf + NTA;
    float* old_logprobs_buf = buf + NTA + NT;
    float* advantages_buf = buf + NTA + NT * 2;
    float* values_buf = buf + NTA + NT * 3;
    float* returns_buf = buf + NTA + NT * 4;
    float* prio_buf = buf + NTA + NT * 5;

    double* actions_buf = (double*)malloc(NT * sizeof(double));

    float adv_sum = 0.0f, adv_sq_sum = 0.0f;
    for (int i = 0; i < NT; ++i) {
        advantages_buf[i] = rand1();
        adv_sum += advantages_buf[i];
        adv_sq_sum += advantages_buf[i] * advantages_buf[i];
    }
    float adv_mean = adv_sum / NT;
    float adv_var = adv_sq_sum / NT - adv_mean * adv_mean;

    for (int i = 0; i < NTA; ++i) {
        logits_buf[i] = rand1() * 2.0f;
    }
    for (int i = 0; i < NT; ++i) {
        values_pred_buf[i] = rand1();
        actions_buf[i] = (double)(rand() % actions);
        old_logprobs_buf[i] = rand1() * 2.0f;
        values_buf[i] = rand1();
        returns_buf[i] = rand1();
    }
    for (int i = 0; i < batch; ++i) {
        prio_buf[i] = (float)rand() / RAND_MAX;
    }

    float_to_device(args->logits, logits_buf, NTA);
    float_to_device(args->values_pred, values_pred_buf, NT);
    cudaMemcpy(args->actions, actions_buf, NT * sizeof(double), cudaMemcpyHostToDevice);
    float_to_device(args->old_logprobs, old_logprobs_buf, NT);
    cudaMemcpy(args->advantages, advantages_buf, NT * sizeof(float), cudaMemcpyHostToDevice);
    float_to_device(args->prio, prio_buf, batch);
    float_to_device(args->values, values_buf, NT);
    float_to_device(args->returns, returns_buf, NT);
    cudaMemcpy(args->adv_mean, &adv_mean, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->adv_var, &adv_var, sizeof(float), cudaMemcpyHostToDevice);

    float grad_loss_val = 1.0f;
    cudaMemcpy(args->grad_loss, &grad_loss_val, sizeof(float), cudaMemcpyHostToDevice);

    args->clip_coef = 0.1f;
    args->vf_clip_coef = 0.1f;
    args->vf_coef = 0.5f;
    args->ent_coef = 0.01f;

    args->logits_stride_n = seq * actions;
    args->logits_stride_t = actions;
    args->logits_stride_a = 1;
    args->values_stride_n = seq;
    args->values_stride_t = 1;

    free(buf);
    free(actions_buf);
    return args;
}

void free_ppolossargs(PPOLossArgs* args) {
    cudaFree(args->logits);
    cudaFree(args->values_pred);
    cudaFree(args->actions);
    cudaFree(args->old_logprobs);
    cudaFree(args->advantages);
    cudaFree(args->prio);
    cudaFree(args->values);
    cudaFree(args->returns);
    cudaFree(args->adv_mean);
    cudaFree(args->adv_var);
    cudaFree(args->loss);
    cudaFree(args->losses_acc);
    cudaFree(args->saved_for_backward);
    cudaFree(args->ratio_out);
    cudaFree(args->newvalue_out);
    cudaFree(args->grad_logits);
    cudaFree(args->grad_values_pred);
    cudaFree(args->grad_loss);
    cudaFree(args->act_sizes);
    free(args);
}

void run_ppoloss_forward(PPOLossArgs* args) {
    int total = args->N * args->T;
    int ppo_grid = (total + PPO_THREADS - 1) / PPO_THREADS;
    cudaMemset(args->loss, 0, sizeof(float));

    // Allocate partials buffer for deterministic reduction
    static float* ppo_partials_buf = nullptr;
    static int ppo_partials_cap = 0;
    int needed = ppo_grid * (LOSS_N + 1);
    if (!ppo_partials_buf || needed > ppo_partials_cap) {
        if (ppo_partials_buf) cudaFree(ppo_partials_buf);
        ppo_partials_cap = needed;
        cudaMalloc(&ppo_partials_buf, ppo_partials_cap * sizeof(float));
    }

    ppo_loss_forward_kernel_optimized<<<ppo_grid, PPO_THREADS>>>(
        args->loss, args->losses_acc,
        ppo_partials_buf,
        args->saved_for_backward,
        args->ratio_out, args->newvalue_out,
        args->logits,
        nullptr,  // logstd (nullptr for discrete)
        args->values_pred, args->actions,
        args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, args->adv_mean, args->adv_var,
        args->act_sizes, args->num_atns,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef,
        args->T, args->A, args->N,
        args->logits_stride_n, args->logits_stride_t, args->logits_stride_a,
        args->values_stride_n, args->values_stride_t,
        false);  // is_continuous

    ppo_loss_reduce_kernel<<<1, LOSS_N + 1>>>(
        args->loss, args->losses_acc, ppo_partials_buf, ppo_grid);
}

void run_ppoloss_backward(PPOLossArgs* args) {
    int total = args->N * args->T;
    int ppo_grid = (total + PPO_THREADS - 1) / PPO_THREADS;
    ppo_loss_backward_kernel_optimized<<<ppo_grid, PPO_THREADS>>>(
        args->grad_logits,
        nullptr,  // grad_logstd (nullptr for discrete)
        args->grad_values_pred, args->grad_loss,
        args->logits,
        nullptr,  // logstd (nullptr for discrete)
        args->values_pred, args->actions,
        args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, args->adv_mean, args->adv_var,
        args->act_sizes, args->num_atns,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef,
        args->T, args->A, args->N,
        args->logits_stride_n, args->logits_stride_t, args->logits_stride_a,
        args->values_stride_n, args->values_stride_t,
        false);  // is_continuous
}

void profile_ppoloss(int batch, int seq, int actions) {
    PPOLossArgs* args = create_ppolossargs(batch, seq, actions);

    int NT = batch*seq;
    printf("ppo_loss (NT=%d, %dx%d, A=%d)\n", NT, batch, seq, actions);

    float fwd_ms = profile_kernel((kernel_fn)run_ppoloss_forward, args);
    print_timing("forward", fwd_ms, NT);

    float bwd_ms = profile_kernel((kernel_fn)run_ppoloss_backward, args);
    print_timing("backward", bwd_ms, NT);

#ifdef USE_TORCH
    torch_bench_ppoloss(args);
#endif
    printf("\n");

    free_ppolossargs(args);
}

// ============================================================================
// Section 7: Sample logits — raw kernel create/free/run
// ============================================================================

SampleLogitsArgs* create_samplelogitsargs(int batch, int num_actions) {
    SampleLogitsArgs* args = (SampleLogitsArgs*)calloc(1, sizeof(SampleLogitsArgs));
    args->B = batch;
    args->A = num_actions;
    args->seed = 42;

    int N_logits = batch * num_actions;
    int N_batch = batch;

    cudaMalloc(&args->logits, N_logits * sizeof(precision_t));
    cudaMalloc(&args->value, N_batch * sizeof(precision_t));
    cudaMalloc(&args->actions, N_batch * sizeof(double));
    cudaMalloc(&args->logprobs, N_batch * sizeof(precision_t));
    cudaMalloc(&args->value_out, N_batch * sizeof(precision_t));
    cudaMalloc(&args->offset, sizeof(int64_t));
    cudaMalloc(&args->act_sizes, sizeof(int));
    cudaMemset(args->offset, 0, sizeof(int64_t));
    cudaMemcpy(args->act_sizes, &num_actions, sizeof(int), cudaMemcpyHostToDevice);

    float* logits_buf = (float*)malloc(N_logits * sizeof(float));
    float* value_buf = (float*)malloc(N_batch * sizeof(float));

    for (int i = 0; i < N_logits; ++i) {
        logits_buf[i] = rand1() * 5.0f;
    }
    for (int i = 0; i < N_batch; ++i) {
        value_buf[i] = rand1();
    }

    float_to_device(args->logits, logits_buf, N_logits);
    float_to_device(args->value, value_buf, N_batch);

    free(logits_buf);
    free(value_buf);
    return args;
}

void free_samplelogitsargs(SampleLogitsArgs* args) {
    cudaFree(args->logits);
    cudaFree(args->value);
    cudaFree(args->actions);
    cudaFree(args->logprobs);
    cudaFree(args->value_out);
    cudaFree(args->offset);
    cudaFree(args->act_sizes);
    free(args);
}

void run_samplelogits_forward(SampleLogitsArgs* args) {
    sample_logits_kernel<<<grid_size(args->B), BLOCK_SIZE>>>(
        args->actions, args->logprobs, args->value_out,
        args->logits,
        nullptr,  // logstd (nullptr for discrete)
        args->value,
        args->act_sizes, args->seed, args->offset,
        1,  // num_atns
        args->B,
        args->A,  // logits_stride
        0,        // logstd_stride (unused for discrete)
        1,        // value_stride
        false);   // is_continuous
}

void profile_samplelogits(int batch, int num_actions) {
    SampleLogitsArgs* args = create_samplelogitsargs(batch, num_actions);

    printf("sample_logits (B=%d, A=%d)\n", batch, num_actions);

    float fwd_ms = profile_kernel((kernel_fn)run_samplelogits_forward, args);
    print_timing("forward", fwd_ms, batch);

#ifdef USE_TORCH
    torch_bench_samplelogits(args);
#endif
    printf("\n");

    free_samplelogitsargs(args);
}

// ============================================================================
// Section 8: Forward call profiling (torch only — delegated to lib)
// ============================================================================

void profile_forwardcall(int batch, int input_size, int hidden_size, int num_atns, int num_layers) {
#ifdef USE_TORCH
    torch_profile_forwardcall(batch, input_size, hidden_size, num_atns, num_layers);
#else
    printf("forward_call: requires USE_TORCH\n\n");
#endif
}

// ============================================================================
// Section 9: Environment speed profiling
// ============================================================================

#ifdef USE_STATIC_ENV

#include "pufferlib/src/vecenv.h"
#include "pufferlib/src/ini.h"

#ifndef ENV_NAME
#error "ENV_NAME must be defined at compile time (e.g. -DENV_NAME=breakout)"
#endif
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

static void empty_net_callback(void* ctx, int buf, int t) {
    (void)ctx; (void)buf; (void)t;
}

static void empty_thread_init(void* ctx, int buf) {
    (void)ctx; (void)buf;
}

typedef struct {
    StaticVec* vec;
    int num_envs;
    int num_buffers;
    int num_threads;
    int horizon;
    int obs_size;
    int num_atns;
} EnvSpeedArgs;

static int ini_handler_env(void* user, const char* section,
                           const char* name, const char* value) {
    Dict* env_kwargs = (Dict*)user;
    if (strcmp(section, "env") == 0) {
        dict_set(env_kwargs, strdup(name), atof(value));
    }
    return 1;
}

typedef struct { int total_agents; int num_buffers; } VecDefaults;
static int ini_handler_vec(void* user, const char* section,
                           const char* name, const char* value) {
    VecDefaults* defaults = (VecDefaults*)user;
    if (strcmp(section, "vec") == 0) {
        if (strcmp(name, "total_agents") == 0) defaults->total_agents = atoi(value);
        else if (strcmp(name, "num_buffers") == 0) defaults->num_buffers = atoi(value);
    }
    return 1;
}

EnvSpeedArgs* create_envspeedargs(int total_agents, int num_buffers, int num_threads, int horizon) {
    char ini_path[512];
    snprintf(ini_path, sizeof(ini_path), "pufferlib/config/ocean/%s.ini", TOSTRING(ENV_NAME));

    VecDefaults defaults = {0};
    if (ini_parse(ini_path, ini_handler_vec, &defaults) < 0) {
        fprintf(stderr, "Warning: Could not load config %s\n", ini_path);
    }

    if (total_agents == 0) total_agents = defaults.total_agents > 0 ? defaults.total_agents : 8192;
    if (num_buffers == 0) num_buffers = defaults.num_buffers > 0 ? defaults.num_buffers : 2;

    Dict* env_kwargs = create_dict(64);
    if (ini_parse(ini_path, ini_handler_env, env_kwargs) < 0) {
        fprintf(stderr, "Warning: Could not load [env] config from %s\n", ini_path);
    }

    Dict* vec_kwargs = create_dict(8);
    dict_set(vec_kwargs, "total_agents", (double)total_agents);
    dict_set(vec_kwargs, "num_buffers", (double)num_buffers);

    StaticVec* vec = create_static_vec(total_agents, num_buffers, vec_kwargs, env_kwargs);
    if (!vec) {
        fprintf(stderr, "Failed to create environments\n");
        return nullptr;
    }
    for (int i = 0; i < num_buffers; i++) {
        cudaStreamCreateWithFlags(&vec->streams[i], cudaStreamNonBlocking);
    }

    int num_envs = vec->size;
    printf("Created %d envs (%s) for %d total_agents\n", num_envs, TOSTRING(ENV_NAME), total_agents);

    create_static_threads(vec, num_threads, horizon, nullptr, empty_net_callback, empty_thread_init);

    static_vec_reset(vec);
    cudaDeviceSynchronize();

    EnvSpeedArgs* args = (EnvSpeedArgs*)calloc(1, sizeof(EnvSpeedArgs));
    args->vec = vec;
    args->num_envs = num_envs;
    args->num_buffers = num_buffers;
    args->num_threads = num_threads;
    args->horizon = horizon;
    args->obs_size = get_obs_size();
    args->num_atns = get_num_atns();

    return args;
}

void free_envspeedargs(EnvSpeedArgs* args) {
    free(args);
}

void run_env_rollout(EnvSpeedArgs* args) {
    static_vec_omp_step(args->vec);
}

float profile_env_rollout(EnvSpeedArgs* args, const char* name) {
    const float ENV_TIMEOUT_SEC = 3.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto start_time = std::chrono::steady_clock::now();
    for (int i = 0; i < 10; ++i) {
        run_env_rollout(args);
        cudaDeviceSynchronize();
        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - start_time).count();
        if (elapsed > ENV_TIMEOUT_SEC) break;
    }

    start_time = get_time_sec();
    cudaProfilerStart();
    if (name) nvtxRangePushA(name);
    cudaEventRecord(start);
    float completed = 0;
    for (int i = 0; i < 1000; ++i) {
        run_env_rollout(args);
        completed += 1;
        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - start_time).count();
        if (elapsed > ENV_TIMEOUT_SEC) break;
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    if (name) nvtxRangePop();
    cudaProfilerStop();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / completed;
}

void profile_envspeed(int total_agents, int num_buffers, int num_threads, int horizon) {
    printf("env_speed_static (total_agents=%d, buffers=%d, threads=%d, horizon=%d)\n",
           total_agents, num_buffers, num_threads, horizon);

    EnvSpeedArgs* args = create_envspeedargs(total_agents, num_buffers, num_threads, horizon);
    if (!args) {
        printf("  Failed to create env - skipping\n\n");
        return;
    }

    printf("  num_envs=%d, obs_size=%d, num_atns=%d\n", args->num_envs, args->obs_size, args->num_atns);

    float rollout_ms = profile_env_rollout(args, "env_rollout");
    int total_steps = total_agents * horizon;
    printf("  rollout time: %.2f ms (%d steps)\n", rollout_ms, total_steps);

    float sps = total_steps / rollout_ms * 1000.0f;
    printf("  throughput: %.2f M steps/s\n", sps / 1e6);

    free_envspeedargs(args);
    printf("\n");
}

#endif  // USE_STATIC_ENV

// ============================================================================
// Section 10: Rollout copy profiling (torch only — delegated to lib)
// ============================================================================

// ============================================================================
// Section 11: Training profiling (torch only — delegated to lib)
// ============================================================================

// ============================================================================
// Section 12: main() dispatcher
// ============================================================================

void print_usage(const char* prog) {
    printf("Usage: %s <profile>\n", prog);
    printf("\nProfiles:\n");
    printf("  kernels        - All individual kernel microbenchmarks\n");
    printf("  mingrugate     - MinGRU gate kernel only\n");
    printf("  logcumsumexp   - Logcumsumexp kernel only\n");
    printf("  fusedscan      - Fused scan (checkpointed) kernel only\n");
    printf("  samplelogits   - Sample logits kernel only\n");
    printf("  ppoloss        - PPO loss kernel only\n");
    printf("  fcmax          - FC+Max kernel only\n");
#ifdef USE_TORCH
    printf("  forwardcall    - Inference forward pass (requires torch)\n");
    printf("  trainforward   - Training fwd+loss+bwd breakdown (requires torch)\n");
    printf("  trainstep      - Full training step with Muon optimizer (requires torch)\n");
    printf("  rolloutcopy    - Per-minibatch data prep: advantage+prio+copy (requires torch)\n");
#endif
#ifdef USE_STATIC_ENV
    printf("  envspeed       - Environment step throughput (static linked)\n");
    printf("    --buffers N  - Number of buffers (default: %d)\n", BUF);
    printf("    --threads N  - Number of threads (default: 16)\n");
    printf("    --horizon N  - Horizon length (default: %d)\n", T);
#endif
    printf("  all            - Run all available profiles\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char* profile = argv[1];

    // Parse optional CLI args for envspeed
    int buffers = BUF;
    int threads = 16;
    int horizon = T;
    int total_agents = BR * buffers;
    for (int i = 2; i < argc - 1; i++) {
        if (strcmp(argv[i], "--buffers") == 0) buffers = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0) threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--horizon") == 0) horizon = atoi(argv[++i]);
        else if (strcmp(argv[i], "--total-agents") == 0) total_agents = atoi(argv[++i]);
    }

    warmup_gpu();

    // Using typical breakout settings: INPUT_SIZE=96, H=128, A=4
    bool run_all = strcmp(profile, "all") == 0;

    // === Individual kernel microbenchmarks ===
    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "mingrugate") == 0 || run_all) {
        profile_mingrugate(BR, H);
    }
    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "logcumsumexp") == 0 || run_all) {
        profile_logcumsumexp(BT, T, H);
    }
    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "fusedscan") == 0 || run_all) {
        profile_fusedscan(BT, T, H);
    }
    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "samplelogits") == 0 || run_all) {
        profile_samplelogits(BR, A);
    }
    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "ppoloss") == 0 || run_all) {
        profile_ppoloss(BT, T, A);
    }
    // TODO: fc_max cuBLAS issue — investigate separately
    // if (strcmp(profile, "kernels") == 0 || strcmp(profile, "fcmax") == 0 || run_all) {
    //     profile_fcmax(BR, 63, 7, 128);    // partner encoder (drive)
    //     profile_fcmax(BR, 200, 13, 128);  // road encoder (drive)
    // }

    // === Composite profiles (require torch) ===
#ifdef USE_TORCH
    if (strcmp(profile, "forwardcall") == 0 || run_all) {
        torch_profile_forwardcall(BR, INPUT_SIZE, H, A, 1);
    }
    if (strcmp(profile, "trainforward") == 0 || run_all) {
        torch_profile_trainforward(BT, T, INPUT_SIZE, H, A, 1);
    }
    if (strcmp(profile, "trainstep") == 0 || run_all) {
        torch_profile_trainstep(BT, T, INPUT_SIZE, H, A, 1);
    }
    if (strcmp(profile, "rolloutcopy") == 0 || run_all) {
        // num_segments = BR*BUF (full rollout), minibatch_segs = BT
        torch_profile_rolloutcopy(BR * BUF, T, BT, INPUT_SIZE, A, 1, H);
    }
#endif

    // === Environment speed (requires static env link) ===
#ifdef USE_STATIC_ENV
    if (strcmp(profile, "envspeed") == 0 || strcmp(profile, "all") == 0) {
        profile_envspeed(total_agents, buffers, threads, horizon);
    }
#endif

    if (!run_all
        && strcmp(profile, "kernels") != 0
        && strcmp(profile, "mingrugate") != 0
        && strcmp(profile, "logcumsumexp") != 0
        && strcmp(profile, "fusedscan") != 0
        && strcmp(profile, "samplelogits") != 0
        && strcmp(profile, "ppoloss") != 0
        && strcmp(profile, "fcmax") != 0
#ifdef USE_TORCH
        && strcmp(profile, "forwardcall") != 0
        && strcmp(profile, "trainforward") != 0
        && strcmp(profile, "trainstep") != 0
        && strcmp(profile, "rolloutcopy") != 0
#endif
#ifdef USE_STATIC_ENV
        && strcmp(profile, "envspeed") != 0
#endif
    ) {
        printf("Unknown profile: %s\n\n", profile);
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
