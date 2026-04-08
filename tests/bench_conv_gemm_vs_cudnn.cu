// Benchmark: gemm_conv_* (im2col + cuBLAS) vs conv_* (cuDNN), same weights/inputs.
// Baseline for correctness: gemm path (per project convention).
//
// Build: see tests/bench_conv_gemm_vs_cudnn.sh (-DPRECISION_FLOAT => fp32; omit => bf16)

#include <string>
#include "models.cu"
#include "ocean.cu"

#ifndef PRECISION_FLOAT
#include <cuda_bf16.h>
#endif
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

static void fill_rand_host(float* h, int n, unsigned s) {
    for (int i = 0; i < n; ++i) {
        s = s * 1103515245u + 12345u;
        h[i] = ((s >> 16) & 0x7fff) / 16384.0f - 1.0f;
    }
}

static void stats_diff(const float* a, const float* b, int n, float* max_abs, float* mean_abs) {
    float mx = 0.0f, sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
        sum += d;
    }
    *max_abs = mx;
    *mean_abs = sum / (float)std::max(1, n);
}

static void copy_precision_d2h(const precision_t* d, int n, std::vector<float>* hf) {
    hf->resize((size_t)n);
#ifdef PRECISION_FLOAT
    cudaMemcpy(hf->data(), d, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
#else
    std::vector<precision_t> h((size_t)n);
    cudaMemcpy(h.data(), d, (size_t)n * sizeof(precision_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) (*hf)[i] = __bfloat162float(h[i]);
#endif
}

static void copy_fp32_h2d(const float* h, precision_t* d, int n) {
#ifdef PRECISION_FLOAT
    cudaMemcpy(d, h, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
#else
    std::vector<precision_t> hb((size_t)n);
    for (int i = 0; i < n; ++i) hb[i] = __float2bfloat16(h[i]);
    cudaMemcpy(d, hb.data(), (size_t)n * sizeof(precision_t), cudaMemcpyHostToDevice);
#endif
}

template<typename F>
static float time_kernel_ms(cudaStream_t stream, F fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) {
        fn(stream);
        cudaStreamSynchronize(stream);
    }
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0, stream);
    for (int i = 0; i < iters; ++i) fn(stream);
    cudaEventRecord(ev1, stream);
    cudaEventSynchronize(ev1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    return ms / (float)iters;
}

struct BenchDims {
    int B;
    int IC, OC, K, S, IH, IW;
    bool relu;
};

static void dims_conv1(BenchDims* d, int B) {
    d->B = B;
    d->IC = N3_C1_IC;
    d->OC = N3_C1_OC;
    d->K = N3_C1_K;
    d->S = N3_C1_S;
    d->IH = N3_MAP_H;
    d->IW = N3_MAP_W;
    d->relu = true;
}

static void dims_conv2(BenchDims* d, int B) {
    d->B = B;
    d->IC = N3_C2_IC;
    d->OC = N3_C2_OC;
    d->K = N3_C2_K;
    d->S = N3_C2_S;
    d->IH = N3_C1_OH;
    d->IW = N3_C1_OW;
    d->relu = false;
}

static int run_forward(const BenchDims& dim, int warmup, int iters, bool cudnn_save_input) {
    ConvWeights cw{};
    conv_init(&cw, dim.IC, dim.OC, dim.K, dim.S, dim.IH, dim.IW, dim.relu);

    Allocator param_alloc{};
    conv_reg_params(&cw, &param_alloc);
    if (alloc_create(&param_alloc) != cudaSuccess) {
        fprintf(stderr, "alloc_create params failed\n");
        return 1;
    }
    uint64_t seed = 42;
    conv_init_weights(&cw, &seed, 0);
    cudaDeviceSynchronize();

    int OH = cw.OH;
    int OW = cw.OW;
    int out_elems = dim.B * dim.OC * OH * OW;
    int in_elems = dim.B * dim.IC * dim.IH * dim.IW;

    Allocator act_g{};
    PrecisionTensor out_g{}, col{}, mm{}, input{};
    int col_rows = dim.B * OH * OW;
    int col_cols = dim.IC * dim.K * dim.K;
    out_g = {.shape = {(int64_t)out_elems}};
    col = {.shape = {col_rows, col_cols}};
    mm = {.shape = {col_rows, dim.OC}};
    input = {.shape = {dim.B, dim.IC, dim.IH, dim.IW}};
    alloc_register(&act_g, &out_g);
    alloc_register(&act_g, &col);
    alloc_register(&act_g, &mm);
    alloc_register(&act_g, &input);
    if (alloc_create(&act_g) != cudaSuccess) {
        fprintf(stderr, "alloc_create gemm acts failed\n");
        return 1;
    }

    std::vector<float> hin(in_elems);
    fill_rand_host(hin.data(), in_elems, 99u);
    copy_fp32_h2d(hin.data(), input.data, in_elems);

    Allocator acts{}, grads{};
    ConvActivations ca{};
    conv_reg_train(&cw, &ca, &acts, &grads, dim.B, n3_cudnn_dtype());
    if (alloc_create(&acts) != cudaSuccess || alloc_create(&grads) != cudaSuccess) {
        fprintf(stderr, "alloc_create cudnn failed\n");
        return 1;
    }
    if (!cudnn_save_input) ca.saved_input.data = nullptr;

    cudaStream_t stream = 0;

    gemm_conv_forward(&cw.w, &cw.b, input.data, out_g.data, col.data, mm.data, dim.B, dim.IC, dim.IH,
        dim.IW, dim.OC, dim.K, dim.S, OH, OW, dim.relu, stream);
    cudaDeviceSynchronize();

    conv_forward(&cw, &ca, input.data, dim.B, stream);
    cudaDeviceSynchronize();

    std::vector<float> hg, hc;
    copy_precision_d2h(out_g.data, out_elems, &hg);
    copy_precision_d2h(ca.out.data, out_elems, &hc);

    float max_abs, mean_abs;
    stats_diff(hg.data(), hc.data(), out_elems, &max_abs, &mean_abs);
    printf("  forward max |diff|:  %.6g   mean |diff|: %.6g\n", max_abs, mean_abs);

    auto run_gemm = [&](cudaStream_t s) {
        gemm_conv_forward(&cw.w, &cw.b, input.data, out_g.data, col.data, mm.data, dim.B, dim.IC,
            dim.IH, dim.IW, dim.OC, dim.K, dim.S, OH, OW, dim.relu, s);
    };
    auto run_cudnn = [&](cudaStream_t s) {
        conv_forward(&cw, &ca, input.data, dim.B, s);
    };

    float ms_g = time_kernel_ms(stream, run_gemm, warmup, iters);
    float ms_c = time_kernel_ms(stream, run_cudnn, warmup, iters);
    printf("  gemm_conv_forward:  %8.4f us/iter\n", ms_g * 1000.0f);
    printf("  conv_forward:       %8.4f us/iter  (%.2fx vs gemm)\n", ms_c * 1000.0f, ms_g / ms_c);

    alloc_free(&param_alloc);
    alloc_free(&act_g);
    alloc_free(&acts);
    alloc_free(&grads);
    return 0;
}

// ∂L/∂W only: gemm path vs cudnnConvolutionBackwardFilter.
// Correctness: gemm_conv_backward(..., input_grad=null) vs conv_backward(..., nullptr).
// Timings: inlined nchw+im2col+mm_tn, gemm_conv_backward, and cudnn BackwardFilter.
static int run_filter_backward_bench(const BenchDims& dim, int warmup, int iters) {
    int B = dim.B, IC = dim.IC, OC = dim.OC, K = dim.K, S = dim.S, IH = dim.IH, IW = dim.IW;
    int OH = (IH - K) / S + 1;
    int OW = (IW - K) / S + 1;
    int col_rows = B * OH * OW;
    int col_cols = IC * K * K;
    int total_col = col_rows * col_cols;
    int total_out = B * OC * OH * OW;
    int in_elems = B * IC * IH * IW;
    int spatial = OH * OW;
    int w_elems = OC * col_cols;

    ConvWeights cw{};
    conv_init(&cw, IC, OC, K, S, IH, IW, false);
    Allocator param_alloc{};
    conv_reg_params(&cw, &param_alloc);
    if (alloc_create(&param_alloc) != cudaSuccess) return 1;
    uint64_t seed = 301;
    conv_init_weights(&cw, &seed, 0);
    cudaDeviceSynchronize();

    Allocator act{};
    PrecisionTensor col{}, mm{}, saved_in{}, grad_out{}, wgrad{};
    col = {.shape = {col_rows, col_cols}};
    mm = {.shape = {col_rows, OC}};
    saved_in = {.shape = {B, IC, IH, IW}};
    grad_out = {.shape = {(int64_t)total_out}};
    wgrad = {.shape = {OC, col_cols}};
    alloc_register(&act, &col);
    alloc_register(&act, &mm);
    alloc_register(&act, &saved_in);
    alloc_register(&act, &grad_out);
    alloc_register(&act, &wgrad);
    if (alloc_create(&act) != cudaSuccess) return 1;

    Allocator acts{}, grads{};
    ConvActivations ca{};
    conv_reg_train(&cw, &ca, &acts, &grads, B, n3_cudnn_dtype());
    if (alloc_create(&acts) != cudaSuccess || alloc_create(&grads) != cudaSuccess) return 1;

    std::vector<float> hs(in_elems), hg(total_out);
    fill_rand_host(hs.data(), in_elems, 301u);
    fill_rand_host(hg.data(), total_out, 302u);
    copy_fp32_h2d(hs.data(), saved_in.data, in_elems);
    copy_fp32_h2d(hg.data(), grad_out.data, total_out);
    cudaMemcpy(ca.saved_input.data, saved_in.data, (size_t)in_elems * sizeof(precision_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(ca.grad.data, grad_out.data, (size_t)total_out * sizeof(precision_t), cudaMemcpyDeviceToDevice);

    PrecisionTensor mm_t = {.data = mm.data, .shape = {col_rows, OC}};
    PrecisionTensor col_t = {.data = col.data, .shape = {col_rows, col_cols}};
    PrecisionTensor wg_t = {.data = wgrad.data, .shape = {OC, col_cols}};

    cudaStream_t stream = 0;

    cudaMemset(wgrad.data, 0, (size_t)w_elems * sizeof(precision_t));
    gemm_conv_backward(&cw.w, saved_in.data, grad_out.data, wgrad.data, nullptr, col.data, mm.data, B, IC, IH, IW,
        OC, K, S, OH, OW, stream);
    cudaDeviceSynchronize();
    std::vector<float> h_wg_g, h_wg_c;
    copy_precision_d2h(wgrad.data, w_elems, &h_wg_g);

    cudaMemset(ca.wgrad.data, 0, (size_t)w_elems * sizeof(precision_t));
    conv_backward(&cw, &ca, nullptr, B, stream);
    cudaDeviceSynchronize();
    copy_precision_d2h(ca.wgrad.data, w_elems, &h_wg_c);

    float f_wg_max, f_wg_mean;
    stats_diff(h_wg_g.data(), h_wg_c.data(), w_elems, &f_wg_max, &f_wg_mean);
    printf("  filter ∂W max |diff| (gemm_conv_backward vs cudnn): %.6g   mean |diff|: %.6g\n", f_wg_max,
        f_wg_mean);

    float ms_nchw = time_kernel_ms(
        stream,
        [&](cudaStream_t s) {
            nchw_to_rows_kernel<<<grid_size(total_out), BLOCK_SIZE, 0, s>>>(
                grad_out.data, mm.data, B, OC, spatial);
        },
        warmup, iters);

    float ms_im2col = time_kernel_ms(
        stream,
        [&](cudaStream_t s) {
            im2col_kernel<<<grid_size(total_col), BLOCK_SIZE, 0, s>>>(
                saved_in.data, col.data, B, IC, IH, IW, K, S, OH, OW);
        },
        warmup, iters);

    nchw_to_rows_kernel<<<grid_size(total_out), BLOCK_SIZE, 0, stream>>>(
        grad_out.data, mm.data, B, OC, spatial);
    im2col_kernel<<<grid_size(total_col), BLOCK_SIZE, 0, stream>>>(
        saved_in.data, col.data, B, IC, IH, IW, K, S, OH, OW);
    cudaStreamSynchronize(stream);

    float ms_tn = time_kernel_ms(
        stream,
        [&](cudaStream_t s) { puf_mm_tn(&mm_t, &col_t, &wg_t, s); }, warmup, iters);

    float ms_gemm_chain = time_kernel_ms(
        stream,
        [&](cudaStream_t s) {
            nchw_to_rows_kernel<<<grid_size(total_out), BLOCK_SIZE, 0, s>>>(
                grad_out.data, mm.data, B, OC, spatial);
            im2col_kernel<<<grid_size(total_col), BLOCK_SIZE, 0, s>>>(
                saved_in.data, col.data, B, IC, IH, IW, K, S, OH, OW);
            puf_mm_tn(&mm_t, &col_t, &wg_t, s);
        },
        warmup, iters);

    float ms_gemm_conv_bwd_wonly = time_kernel_ms(
        stream,
        [&](cudaStream_t s) {
            cudaMemset(wgrad.data, 0, (size_t)w_elems * sizeof(precision_t));
            gemm_conv_backward(&cw.w, saved_in.data, grad_out.data, wgrad.data, nullptr, col.data, mm.data, B, IC,
                IH, IW, OC, K, S, OH, OW, s);
        },
        warmup, iters);

    float ms_cudnn_filt = time_kernel_ms(
        stream,
        [&](cudaStream_t s) {
            cudaMemset(ca.wgrad.data, 0, (size_t)w_elems * sizeof(precision_t));
            conv_backward(&cw, &ca, nullptr, B, s);
        },
        warmup, iters);

    float sum_iso = (ms_nchw + ms_im2col + ms_tn) * 1000.0f;
    printf("  gemm: nchw_to_rows only:     %8.4f us/iter\n", ms_nchw * 1000.0f);
    printf("  gemm: im2col only:           %8.4f us/iter\n", ms_im2col * 1000.0f);
    printf("  gemm: puf_mm_tn only:       %8.4f us/iter  (mm/col prefilled)\n", ms_tn * 1000.0f);
    printf("  gemm: nchw+im2col+mm_tn:     %8.4f us/iter  (chained, matches ∂W slice)\n", ms_gemm_chain * 1000.0f);
    printf("  gemm: gemm_conv_backward:    %8.4f us/iter  (input_grad=null, same as ocean.cu ∂W-only)\n",
        ms_gemm_conv_bwd_wonly * 1000.0f);
    printf("  sum(3 isolated):             %8.4f us/iter  (vs chained)\n", sum_iso);
    printf("  cudnn: BackwardFilter only: %8.4f us/iter  (conv_backward, input_grad=null)\n",
        ms_cudnn_filt * 1000.0f);
    printf("  ratio gemm_chain/cudnn:      %.2fx  (>1 => cudnn faster)\n", ms_gemm_chain / ms_cudnn_filt);
    printf("  ratio gemm_conv_bwd/cudnn:   %.2fx\n", ms_gemm_conv_bwd_wonly / ms_cudnn_filt);

    alloc_free(&param_alloc);
    alloc_free(&act);
    alloc_free(&acts);
    alloc_free(&grads);
    return 0;
}

static int run_backward(const BenchDims& dim, int warmup, int iters) {
    ConvWeights cw{};
    conv_init(&cw, dim.IC, dim.OC, dim.K, dim.S, dim.IH, dim.IW, false);

    Allocator param_alloc{};
    conv_reg_params(&cw, &param_alloc);
    if (alloc_create(&param_alloc) != cudaSuccess) return 1;
    uint64_t seed = 7;
    conv_init_weights(&cw, &seed, 0);
    cudaDeviceSynchronize();

    int OH = cw.OH;
    int OW = cw.OW;
    int out_elems = dim.B * dim.OC * OH * OW;
    int in_elems = dim.B * dim.IC * dim.IH * dim.IW;
    int w_elems = (int)numel(cw.w.shape);

    Allocator act_g{};
    PrecisionTensor col{}, mm{};
    int col_rows = dim.B * OH * OW;
    int col_cols = dim.IC * dim.K * dim.K;
    col = {.shape = {col_rows, col_cols}};
    mm = {.shape = {col_rows, dim.OC}};
    PrecisionTensor saved_in{}, grad_out{}, wgrad_g{};
    saved_in = {.shape = {dim.B, dim.IC, dim.IH, dim.IW}};
    grad_out = {.shape = {(int64_t)out_elems}};
    wgrad_g = {.shape = {cw.w.shape[0], cw.w.shape[1]}};
    PrecisionTensor dinput_g{};
    dinput_g = {.shape = {dim.B, dim.IC, dim.IH, dim.IW}};
    alloc_register(&act_g, &col);
    alloc_register(&act_g, &mm);
    alloc_register(&act_g, &saved_in);
    alloc_register(&act_g, &grad_out);
    alloc_register(&act_g, &wgrad_g);
    alloc_register(&act_g, &dinput_g);
    if (alloc_create(&act_g) != cudaSuccess) return 1;

    std::vector<float> hs(in_elems), hg(out_elems);
    fill_rand_host(hs.data(), in_elems, 101u);
    fill_rand_host(hg.data(), out_elems, 202u);
    copy_fp32_h2d(hs.data(), saved_in.data, in_elems);
    copy_fp32_h2d(hg.data(), grad_out.data, out_elems);

    Allocator acts{}, grads{};
    ConvActivations ca{};
    conv_reg_train(&cw, &ca, &acts, &grads, dim.B, n3_cudnn_dtype());
    if (alloc_create(&acts) != cudaSuccess || alloc_create(&grads) != cudaSuccess) return 1;
    cudaMemcpy(ca.saved_input.data, saved_in.data, (size_t)in_elems * sizeof(precision_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(ca.grad.data, grad_out.data, (size_t)out_elems * sizeof(precision_t), cudaMemcpyDeviceToDevice);

    cudaStream_t stream = 0;

    cudaMemset(wgrad_g.data, 0, (size_t)w_elems * sizeof(precision_t));
    cudaMemset(dinput_g.data, 0, (size_t)in_elems * sizeof(precision_t));
    gemm_conv_backward(&cw.w, saved_in.data, grad_out.data, wgrad_g.data, dinput_g.data, col.data, mm.data,
        dim.B, dim.IC, dim.IH, dim.IW, dim.OC, dim.K, dim.S, OH, OW, stream);
    cudaDeviceSynchronize();

    std::vector<float> hwg, hdi;
    copy_precision_d2h(wgrad_g.data, w_elems, &hwg);
    copy_precision_d2h(dinput_g.data, in_elems, &hdi);

    cudaMemset(ca.wgrad.data, 0, (size_t)w_elems * sizeof(precision_t));
    cudaMemset(dinput_g.data, 0, (size_t)in_elems * sizeof(precision_t));
    conv_backward(&cw, &ca, dinput_g.data, dim.B, stream);
    cudaDeviceSynchronize();

    std::vector<float> hwg_c, hdi_c;
    copy_precision_d2h(ca.wgrad.data, w_elems, &hwg_c);
    copy_precision_d2h(dinput_g.data, in_elems, &hdi_c);

    float mw, mdw, mdi, mdi_mean;
    stats_diff(hwg.data(), hwg_c.data(), w_elems, &mw, &mdw);
    stats_diff(hdi.data(), hdi_c.data(), in_elems, &mdi, &mdi_mean);
    printf("  backward wgrad max |diff|:   %.6g\n", mw);
    printf("  backward d_input max |diff|: %.6g\n", mdi);

    auto run_gemm_b = [&](cudaStream_t s) {
        cudaMemset(wgrad_g.data, 0, (size_t)w_elems * sizeof(precision_t));
        cudaMemset(dinput_g.data, 0, (size_t)in_elems * sizeof(precision_t));
        gemm_conv_backward(&cw.w, saved_in.data, grad_out.data, wgrad_g.data, dinput_g.data, col.data,
            mm.data, dim.B, dim.IC, dim.IH, dim.IW, dim.OC, dim.K, dim.S, OH, OW, s);
    };
    auto run_cudnn_b = [&](cudaStream_t s) {
        cudaMemset(ca.wgrad.data, 0, (size_t)w_elems * sizeof(precision_t));
        cudaMemset(dinput_g.data, 0, (size_t)in_elems * sizeof(precision_t));
        conv_backward(&cw, &ca, dinput_g.data, dim.B, s);
    };

    float ms_g = time_kernel_ms(stream, run_gemm_b, warmup, iters);
    float ms_c = time_kernel_ms(stream, run_cudnn_b, warmup, iters);
    printf("  gemm_conv_backward:  %8.4f us/iter\n", ms_g * 1000.0f);
    printf("  conv_backward:       %8.4f us/iter  (%.2fx vs gemm)\n", ms_c * 1000.0f, ms_g / ms_c);

    alloc_free(&param_alloc);
    alloc_free(&act_g);
    alloc_free(&acts);
    alloc_free(&grads);
    return 0;
}

static int run_im2col_bench(const BenchDims& dim, int warmup, int iters) {
    int B = dim.B, IC = dim.IC, IH = dim.IH, IW = dim.IW, K = dim.K, S = dim.S;
    int OH = (IH - K) / S + 1;
    int OW = (IW - K) / S + 1;
    int total_col = B * OH * OW * IC * K * K;
    int in_elems = B * IC * IH * IW;

    precision_t *d_in = nullptr, *d_col_slow = nullptr, *d_col_fast = nullptr;
    if (cudaMalloc(&d_in, (size_t)in_elems * sizeof(precision_t)) != cudaSuccess) return 1;
    if (cudaMalloc(&d_col_slow, (size_t)total_col * sizeof(precision_t)) != cudaSuccess) return 1;
    if (cudaMalloc(&d_col_fast, (size_t)total_col * sizeof(precision_t)) != cudaSuccess) return 1;

    std::vector<float> h_in((size_t)in_elems);
    fill_rand_host(h_in.data(), in_elems, 401u);
    copy_fp32_h2d(h_in.data(), d_in, in_elems);
    cudaDeviceSynchronize();

    const int oh_ow = OH * OW;
    const int col_cols = IC * K * K;
    const int total_no_batch = oh_ow * col_cols;
    const int kk = K * K;
    FastDivMod dm_col_w(col_cols);
    FastDivMod dm_oh_ow(oh_ow);
    FastDivMod dm_ow(OW);
    FastDivMod dm_kk(kk);
    FastDivMod dm_k(K);

    cudaStream_t stream = 0;
    im2col_kernel<<<grid_size(total_col), BLOCK_SIZE, 0, stream>>>(
        d_in, d_col_slow, B, IC, IH, IW, K, S, OH, OW);
    im2col_kernel_fast<<<grid_size(total_col), BLOCK_SIZE, 0, stream>>>(
        d_in, d_col_fast, B, IC, IH, IW, K, S, OH, OW,
        dm_col_w, dm_oh_ow, dm_ow, dm_kk, dm_k, total_no_batch);
    cudaDeviceSynchronize();

    std::vector<float> hs, hf;
    copy_precision_d2h(d_col_slow, total_col, &hs);
    copy_precision_d2h(d_col_fast, total_col, &hf);
    float max_d = 0.0f, mean_d = 0.0f;
    stats_diff(hs.data(), hf.data(), total_col, &max_d, &mean_d);
    printf("  im2col vs im2col_fast max |diff|: %.6g  mean |diff|: %.6g\n", max_d, mean_d);

    float ms_slow = time_kernel_ms(
        stream,
        [&](cudaStream_t s) {
            im2col_kernel<<<grid_size(total_col), BLOCK_SIZE, 0, s>>>(
                d_in, d_col_slow, B, IC, IH, IW, K, S, OH, OW);
        },
        warmup, iters);
    float ms_fast = time_kernel_ms(
        stream,
        [&](cudaStream_t s) {
            im2col_kernel_fast<<<grid_size(total_col), BLOCK_SIZE, 0, s>>>(
                d_in, d_col_fast, B, IC, IH, IW, K, S, OH, OW,
                dm_col_w, dm_oh_ow, dm_ow, dm_kk, dm_k, total_no_batch);
        },
        warmup, iters);
    printf("  im2col_kernel:       %8.4f us/iter\n", ms_slow * 1000.0f);
    printf("  im2col_kernel_fast:  %8.4f us/iter  (%.2fx vs slow)\n", ms_fast * 1000.0f,
        ms_slow / ms_fast);

    cudaFree(d_in);
    cudaFree(d_col_slow);
    cudaFree(d_col_fast);
    return 0;
}

// gemm_conv_forward vs gemm_conv_forward_fast; relu on/off (NMMO3 layer geometry only).
static int run_gemm_fast_fwd_bench(const BenchDims& dim, int layer, int warmup, int iters) {
    const Im2ColFastMods& m = (layer == 1) ? kIm2ColModsC1 : kIm2ColModsC2;

    ConvWeights cw{};
    conv_init(&cw, dim.IC, dim.OC, dim.K, dim.S, dim.IH, dim.IW, dim.relu);
    Allocator param_alloc{};
    conv_reg_params(&cw, &param_alloc);
    if (alloc_create(&param_alloc) != cudaSuccess) return 1;
    uint64_t seed = 55;
    conv_init_weights(&cw, &seed, 0);
    cudaDeviceSynchronize();

    int OH = cw.OH;
    int OW = cw.OW;
    int out_elems = dim.B * dim.OC * OH * OW;
    int in_elems = dim.B * dim.IC * dim.IH * dim.IW;
    int col_rows = dim.B * OH * OW;
    int col_cols = dim.IC * dim.K * dim.K;

    Allocator act{};
    PrecisionTensor out_s{}, out_f{}, col{}, mm{}, input{};
    out_s = {.shape = {(int64_t)out_elems}};
    out_f = {.shape = {(int64_t)out_elems}};
    col = {.shape = {col_rows, col_cols}};
    mm = {.shape = {col_rows, dim.OC}};
    input = {.shape = {dim.B, dim.IC, dim.IH, dim.IW}};
    alloc_register(&act, &out_s);
    alloc_register(&act, &out_f);
    alloc_register(&act, &col);
    alloc_register(&act, &mm);
    alloc_register(&act, &input);
    if (alloc_create(&act) != cudaSuccess) return 1;

    std::vector<float> hin(in_elems);
    fill_rand_host(hin.data(), in_elems, 77u);
    copy_fp32_h2d(hin.data(), input.data, in_elems);
    cudaDeviceSynchronize();

    cudaStream_t stream = 0;

    for (int ri = 0; ri < 2; ++ri) {
        bool use_relu = (ri == 1);
        gemm_conv_forward(&cw.w, &cw.b, input.data, out_s.data, col.data, mm.data, dim.B, dim.IC, dim.IH,
            dim.IW, dim.OC, dim.K, dim.S, OH, OW, use_relu, stream);
        cudaDeviceSynchronize();
        gemm_conv_forward_fast(&cw.w, &cw.b, input.data, out_f.data, col.data, mm.data, dim.B, m, use_relu,
            stream);
        cudaDeviceSynchronize();
        std::vector<float> hs, hf;
        copy_precision_d2h(out_s.data, out_elems, &hs);
        copy_precision_d2h(out_f.data, out_elems, &hf);
        float mx, mn;
        stats_diff(hs.data(), hf.data(), out_elems, &mx, &mn);
        printf("  relu=%d  max |diff| slow vs fast: %.6g  mean |diff|: %.6g\n", (int)use_relu, mx, mn);

        float ms_slow = time_kernel_ms(
            stream,
            [&](cudaStream_t s) {
                gemm_conv_forward(&cw.w, &cw.b, input.data, out_s.data, col.data, mm.data, dim.B, dim.IC,
                    dim.IH, dim.IW, dim.OC, dim.K, dim.S, OH, OW, use_relu, s);
            },
            warmup, iters);
        float ms_fast = time_kernel_ms(
            stream,
            [&](cudaStream_t s) {
                gemm_conv_forward_fast(&cw.w, &cw.b, input.data, out_f.data, col.data, mm.data, dim.B, m,
                    use_relu, s);
            },
            warmup, iters);
        printf("  relu=%d  gemm_conv_forward:      %8.4f us/iter\n", (int)use_relu, ms_slow * 1000.0f);
        printf("  relu=%d  gemm_conv_forward_fast: %8.4f us/iter  (%.2fx vs slow)\n", (int)use_relu,
            ms_fast * 1000.0f, ms_slow / ms_fast);
    }

    alloc_free(&param_alloc);
    alloc_free(&act);
    return 0;
}

int main(int argc, char** argv) {
    int B = 1024;
    int layer = 1;
    int warmup = 50;
    int iters = 200;
    bool do_fwd = true;
    bool do_bwd = true;
    bool cudnn_save = true;
    bool do_wgrad_breakdown = false;
    bool do_im2col_bench = false;
    bool do_gemm_fast_bench = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-B") == 0 && i + 1 < argc) B = atoi(argv[++i]);
        else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) layer = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--forward-only") == 0) do_bwd = false;
        else if (strcmp(argv[i], "--backward-only") == 0) do_fwd = false;
        else if (strcmp(argv[i], "--no-cudnn-save-input") == 0) cudnn_save = false;
        else if (strcmp(argv[i], "--wgrad-breakdown-only") == 0
            || strcmp(argv[i], "--filter-bwd-only") == 0) {
            do_wgrad_breakdown = true;
            do_fwd = false;
            do_bwd = false;
        } else if (strcmp(argv[i], "--wgrad-breakdown") == 0 || strcmp(argv[i], "--filter-bwd") == 0) {
            do_wgrad_breakdown = true;
        } else if (strcmp(argv[i], "--im2col-bench-only") == 0) {
            do_im2col_bench = true;
            do_fwd = false;
            do_bwd = false;
            do_wgrad_breakdown = false;
        } else if (strcmp(argv[i], "--im2col-bench") == 0) {
            do_im2col_bench = true;
        } else if (strcmp(argv[i], "--gemm-fast-bench-only") == 0) {
            do_gemm_fast_bench = true;
            do_fwd = false;
            do_bwd = false;
            do_wgrad_breakdown = false;
            do_im2col_bench = false;
        } else if (strcmp(argv[i], "--gemm-fast-bench") == 0) {
            do_gemm_fast_bench = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  -B N                 batch size (default 1024)\n");
            printf("  --layer 1|2         NMMO3 conv1 or conv2 sizes (default 1)\n");
            printf("  --warmup N          timing warmup runs (default 50)\n");
            printf("  --iters N           timed iterations (default 200)\n");
            printf("  --forward-only      only forward pass\n");
            printf("  --backward-only     only backward (identity activation)\n");
            printf("  --no-cudnn-save-input  omit cudnn memcpy to saved_input (forward timing)\n");
            printf("  --filter-bwd / --wgrad-breakdown  also bench ∂W: gemm (nchw+im2col+mm_tn) vs cudnn BackwardFilter\n");
            printf("  --filter-bwd-only / --wgrad-breakdown-only  only that ∂W bench\n");
            printf("  --im2col-bench       also bench im2col_kernel vs im2col_kernel_fast\n");
            printf("  --im2col-bench-only  only that im2col bench\n");
            printf("  --gemm-fast-bench    also bench gemm_conv_forward vs gemm_conv_forward_fast (relu 0/1)\n");
            printf("  --gemm-fast-bench-only  only that bench\n");
            printf("  (script) --float / --fp32   compile fp32 (default)\n");
            printf("  (script) --bf16 / --half    compile bf16 (matches default native backend)\n");
            return 0;
        }
    }

    BenchDims dim{};
    if (layer == 1) dims_conv1(&dim, B);
    else if (layer == 2) dims_conv2(&dim, B);
    else {
        fprintf(stderr, "layer must be 1 or 2\n");
        return 1;
    }

    int OH = (dim.IH - dim.K) / dim.S + 1;
    int OW = (dim.IW - dim.K) / dim.S + 1;
    printf("bench_conv_gemm_vs_cudnn  B=%d  layer=%d  IC=%d OC=%d  %dx%d K=%d S=%d -> %dx%d  relu=%d",
        dim.B, layer, dim.IC, dim.OC, dim.IH, dim.IW, dim.K, dim.S, OH, OW, (int)dim.relu);
#ifdef PRECISION_FLOAT
    printf("  precision=fp32\n");
#else
    printf("  precision=bf16\n");
#endif

    if (do_fwd) {
        printf("\n--- forward (gemm baseline vs cudnn) ---\n");
        if (run_forward(dim, warmup, iters, cudnn_save)) return 1;
    }
    if (do_bwd) {
        BenchDims bd = dim;
        bd.relu = false;
        printf("\n--- backward (identity conv; gemm vs cudnn) ---\n");
        printf("  (relu ignored: identity conv so cudnn bwd matches gemm without ReLU mask)\n");
        if (run_backward(bd, warmup, iters)) return 1;
    }
    if (do_wgrad_breakdown) {
        printf("\n--- filter backward (∂W): gemm path vs cudnnConvolutionBackwardFilter ---\n");
        if (run_filter_backward_bench(dim, warmup, iters)) return 1;
    }
    if (do_im2col_bench) {
        printf("\n--- im2col_kernel vs im2col_kernel_fast ---\n");
        if (run_im2col_bench(dim, warmup, iters)) return 1;
    }
    if (do_gemm_fast_bench) {
        printf("\n--- gemm_conv_forward vs gemm_conv_forward_fast (relu off/on) ---\n");
        if (run_gemm_fast_fwd_bench(dim, layer, warmup, iters)) return 1;
    }
    return 0;
}
