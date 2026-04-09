// End-to-end: gemm conv (slow) vs gemm_fast vs cudnn — forward & backward timed separately, layers 1 & 2 (NMMO3).
// Build/run: tests/bench_gemm_conv_end2end.sh [--float|--bf16]

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

// max_i |a-b| / max(|a|, eps) — scale-free compare vs reference `a`.
static float stats_rel_max(const float* a, const float* b, int n, float eps) {
    float mx = 0.0f;
    for (int i = 0; i < n; ++i) {
        float den = fmaxf(fabsf(a[i]), eps);
        float r = fabsf(a[i] - b[i]) / den;
        if (r > mx) mx = r;
    }
    return mx;
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

static int run_layer(int layer, int warmup, int iters) {
    BenchDims dim{};
    if (layer == 1) dims_conv1(&dim, 1024);
    else if (layer == 2) dims_conv2(&dim, 1024);
    else return 1;

    const Im2ColFastMods& m = (layer == 1) ? kIm2ColModsC1 : kIm2ColModsC2;
    int OH = (dim.IH - dim.K) / dim.S + 1;
    int OW = (dim.IW - dim.K) / dim.S + 1;
    if (OH != m.OH || OW != m.OW || dim.IC != m.IC || dim.IH != m.IH || dim.IW != m.IW || dim.OC != m.OC
        || dim.K != m.K || dim.S != m.S) {
        fprintf(stderr, "layer %d: dim mismatch vs Im2ColFastMods\n", layer);
        return 1;
    }

    ConvWeights cw{};
    conv_init(&cw, dim.IC, dim.OC, dim.K, dim.S, dim.IH, dim.IW, dim.relu);

    Allocator param_alloc{};
    conv_reg_params(&cw, &param_alloc);
    if (alloc_create(&param_alloc) != cudaSuccess) return 1;
    uint64_t seed = 1000u + (unsigned)layer;
    conv_init_weights(&cw, &seed, 0);
    cudaDeviceSynchronize();

    int B = dim.B;
    int out_elems = B * dim.OC * OH * OW;
    int in_elems = B * dim.IC * dim.IH * dim.IW;
    int w_elems = (int)numel(cw.w.shape);
    int col_rows = B * OH * OW;
    int col_cols = dim.IC * dim.K * dim.K;

    Allocator act_g{};
    PrecisionTensor out_gemm{}, out_fast{}, col{}, mm{};
    PrecisionTensor saved_in{}, grad_out{}, wgrad_g{}, dinput_g{};
    out_gemm = {.shape = {(int64_t)out_elems}};
    out_fast = {.shape = {(int64_t)out_elems}};
    col = {.shape = {col_rows, col_cols}};
    mm = {.shape = {col_rows, dim.OC}};
    saved_in = {.shape = {B, dim.IC, dim.IH, dim.IW}};
    grad_out = {.shape = {(int64_t)out_elems}};
    wgrad_g = {.shape = {cw.w.shape[0], cw.w.shape[1]}};
    dinput_g = {.shape = {B, dim.IC, dim.IH, dim.IW}};
    alloc_register(&act_g, &out_gemm);
    alloc_register(&act_g, &out_fast);
    alloc_register(&act_g, &col);
    alloc_register(&act_g, &mm);
    alloc_register(&act_g, &saved_in);
    alloc_register(&act_g, &grad_out);
    alloc_register(&act_g, &wgrad_g);
    alloc_register(&act_g, &dinput_g);
    if (alloc_create(&act_g) != cudaSuccess) return 1;

    Allocator acts{}, grads{};
    ConvActivations ca{};
    conv_reg_train(&cw, &ca, &acts, &grads, B, n3_cudnn_dtype());
    if (alloc_create(&acts) != cudaSuccess || alloc_create(&grads) != cudaSuccess) return 1;

    std::vector<float> hin(in_elems), hg_up(out_elems);
    fill_rand_host(hin.data(), in_elems, 201u + (unsigned)layer);
    fill_rand_host(hg_up.data(), out_elems, 303u + (unsigned)layer);
    copy_fp32_h2d(hin.data(), saved_in.data, in_elems);
    cudaMemcpy(ca.saved_input.data, saved_in.data, (size_t)in_elems * sizeof(precision_t), cudaMemcpyDeviceToDevice);

    cudaStream_t stream = 0;
    const float rel_eps = 1e-5f;

    printf("layer %d  B=%d  IC=%d OC=%d  %dx%d K=%d S=%d -> %dx%d  relu=%d\n", layer, B, dim.IC, dim.OC, dim.IH,
        dim.IW, dim.K, dim.S, OH, OW, (int)dim.relu);
    printf("  --- correctness (reference = gemm_conv forward/backward) ---\n");

    gemm_conv_forward(&cw.w, &cw.b, saved_in.data, out_gemm.data, col.data, mm.data, B, dim.IC, dim.IH, dim.IW,
        dim.OC, dim.K, dim.S, OH, OW, dim.relu, stream);
    cudaDeviceSynchronize();
    gemm_conv_forward_fast(&cw.w, &cw.b, saved_in.data, out_fast.data, col.data, mm.data, B, m, dim.relu, stream);
    cudaDeviceSynchronize();
    conv_forward(&cw, &ca, saved_in.data, B, stream);
    cudaDeviceSynchronize();

    std::vector<float> h_gemm_o, h_fast_o, h_cdnn_o;
    copy_precision_d2h(out_gemm.data, out_elems, &h_gemm_o);
    copy_precision_d2h(out_fast.data, out_elems, &h_fast_o);
    copy_precision_d2h(ca.out.data, out_elems, &h_cdnn_o);
    float mx, mn, rel;
    stats_diff(h_gemm_o.data(), h_fast_o.data(), out_elems, &mx, &mn);
    rel = stats_rel_max(h_gemm_o.data(), h_fast_o.data(), out_elems, rel_eps);
    printf("  forward  gemm_fast vs gemm:  max|diff| %.6g  mean|diff| %.6g  max rel err %.6g\n", mx, mn, rel);
    stats_diff(h_gemm_o.data(), h_cdnn_o.data(), out_elems, &mx, &mn);
    rel = stats_rel_max(h_gemm_o.data(), h_cdnn_o.data(), out_elems, rel_eps);
    printf("  forward  cudnn vs gemm:      max|diff| %.6g  mean|diff| %.6g  max rel err %.6g\n", mx, mn, rel);

    copy_fp32_h2d(hg_up.data(), grad_out.data, out_elems);
    cudaMemcpy(ca.grad.data, grad_out.data, (size_t)out_elems * sizeof(precision_t), cudaMemcpyDeviceToDevice);

    cudaMemset(wgrad_g.data, 0, (size_t)w_elems * sizeof(precision_t));
    cudaMemset(dinput_g.data, 0, (size_t)in_elems * sizeof(precision_t));
    gemm_conv_backward(&cw.w, saved_in.data, grad_out.data, wgrad_g.data, dinput_g.data, col.data, mm.data, B,
        dim.IC, dim.IH, dim.IW, dim.OC, dim.K, dim.S, OH, OW, stream);
    cudaDeviceSynchronize();
    std::vector<float> hwg_ref, hdi_ref;
    copy_precision_d2h(wgrad_g.data, w_elems, &hwg_ref);
    copy_precision_d2h(dinput_g.data, in_elems, &hdi_ref);

    cudaMemset(wgrad_g.data, 0, (size_t)w_elems * sizeof(precision_t));
    cudaMemset(dinput_g.data, 0, (size_t)in_elems * sizeof(precision_t));
    gemm_conv_backward_fast(&cw.w, saved_in.data, grad_out.data, wgrad_g.data, dinput_g.data, col.data, mm.data, B,
        m, stream);
    cudaDeviceSynchronize();
    std::vector<float> hwg_f, hdi_f;
    copy_precision_d2h(wgrad_g.data, w_elems, &hwg_f);
    copy_precision_d2h(dinput_g.data, in_elems, &hdi_f);
    stats_diff(hwg_ref.data(), hwg_f.data(), w_elems, &mx, &mn);
    rel = stats_rel_max(hwg_ref.data(), hwg_f.data(), w_elems, rel_eps);
    printf("  backward wgrad gemm_fast vs gemm: max|diff| %.6g  mean|diff| %.6g  max rel err %.6g\n", mx, mn, rel);
    stats_diff(hdi_ref.data(), hdi_f.data(), in_elems, &mx, &mn);
    rel = stats_rel_max(hdi_ref.data(), hdi_f.data(), in_elems, rel_eps);
    printf("  backward d_input gemm_fast vs gemm: max|diff| %.6g  mean|diff| %.6g  max rel err %.6g\n", mx, mn, rel);

    cudaMemset(ca.wgrad.data, 0, (size_t)w_elems * sizeof(precision_t));
    cudaMemset(dinput_g.data, 0, (size_t)in_elems * sizeof(precision_t));
    conv_backward(&cw, &ca, dinput_g.data, B, stream);
    cudaDeviceSynchronize();
    std::vector<float> hwg_c, hdi_c;
    copy_precision_d2h(ca.wgrad.data, w_elems, &hwg_c);
    copy_precision_d2h(dinput_g.data, in_elems, &hdi_c);
    stats_diff(hwg_ref.data(), hwg_c.data(), w_elems, &mx, &mn);
    rel = stats_rel_max(hwg_ref.data(), hwg_c.data(), w_elems, rel_eps);
    printf("  backward wgrad cudnn vs gemm:     max|diff| %.6g  mean|diff| %.6g  max rel err %.6g\n", mx, mn, rel);
    stats_diff(hdi_ref.data(), hdi_c.data(), in_elems, &mx, &mn);
    rel = stats_rel_max(hdi_ref.data(), hdi_c.data(), in_elems, rel_eps);
    printf("  backward d_input cudnn vs gemm: max|diff| %.6g  mean|diff| %.6g  max rel err %.6g\n", mx, mn, rel);

    printf("  --- timing (%d warmup / %d iters): forward and backward measured separately ---\n", warmup, iters);

    auto run_gemm_fwd = [&](cudaStream_t s) {
        gemm_conv_forward(&cw.w, &cw.b, saved_in.data, out_gemm.data, col.data, mm.data, B, dim.IC, dim.IH, dim.IW,
            dim.OC, dim.K, dim.S, OH, OW, dim.relu, s);
    };
    auto run_gemm_fast_fwd = [&](cudaStream_t s) {
        gemm_conv_forward_fast(&cw.w, &cw.b, saved_in.data, out_fast.data, col.data, mm.data, B, m, dim.relu, s);
    };
    auto run_cudnn_fwd = [&](cudaStream_t s) { conv_forward(&cw, &ca, saved_in.data, B, s); };

    auto run_gemm_bwd = [&](cudaStream_t s) {
        cudaMemsetAsync(wgrad_g.data, 0, (size_t)w_elems * sizeof(precision_t), s);
        cudaMemsetAsync(dinput_g.data, 0, (size_t)in_elems * sizeof(precision_t), s);
        gemm_conv_backward(&cw.w, saved_in.data, grad_out.data, wgrad_g.data, dinput_g.data, col.data, mm.data, B,
            dim.IC, dim.IH, dim.IW, dim.OC, dim.K, dim.S, OH, OW, s);
    };
    auto run_gemm_fast_bwd = [&](cudaStream_t s) {
        cudaMemsetAsync(wgrad_g.data, 0, (size_t)w_elems * sizeof(precision_t), s);
        cudaMemsetAsync(dinput_g.data, 0, (size_t)in_elems * sizeof(precision_t), s);
        gemm_conv_backward_fast(&cw.w, saved_in.data, grad_out.data, wgrad_g.data, dinput_g.data, col.data, mm.data,
            B, m, s);
    };
    auto run_cudnn_bwd = [&](cudaStream_t s) {
        cudaMemcpyAsync(ca.saved_input.data, saved_in.data, (size_t)in_elems * sizeof(precision_t),
            cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(ca.grad.data, grad_out.data, (size_t)out_elems * sizeof(precision_t), cudaMemcpyDeviceToDevice,
            s);
        cudaMemsetAsync(ca.wgrad.data, 0, (size_t)w_elems * sizeof(precision_t), s);
        cudaMemsetAsync(dinput_g.data, 0, (size_t)in_elems * sizeof(precision_t), s);
        conv_backward(&cw, &ca, dinput_g.data, B, s);
    };

    float ms_gf = time_kernel_ms(stream, run_gemm_fwd, warmup, iters);
    float ms_ff = time_kernel_ms(stream, run_gemm_fast_fwd, warmup, iters);
    float ms_cf = time_kernel_ms(stream, run_cudnn_fwd, warmup, iters);
    float ms_gb = time_kernel_ms(stream, run_gemm_bwd, warmup, iters);
    float ms_fb = time_kernel_ms(stream, run_gemm_fast_bwd, warmup, iters);
    float ms_cb = time_kernel_ms(stream, run_cudnn_bwd, warmup, iters);

    printf("  forward:\n");
    printf("    gemm (slow): %8.4f us/iter\n", ms_gf * 1000.0f);
    printf("    gemm_fast:   %8.4f us/iter  (%.2fx vs gemm)\n", ms_ff * 1000.0f, ms_gf / ms_ff);
    printf("    cudnn:       %8.4f us/iter  (%.2fx vs gemm, %.2fx vs gemm_fast)\n", ms_cf * 1000.0f, ms_gf / ms_cf,
        ms_ff / ms_cf);
    printf("  backward:\n");
    printf("    gemm (slow): %8.4f us/iter\n", ms_gb * 1000.0f);
    printf("    gemm_fast:   %8.4f us/iter  (%.2fx vs gemm)\n", ms_fb * 1000.0f, ms_gb / ms_fb);
    printf("    cudnn:       %8.4f us/iter  (%.2fx vs gemm, %.2fx vs gemm_fast)\n", ms_cb * 1000.0f, ms_gb / ms_cb,
        ms_fb / ms_cb);
    printf("\n");

    alloc_free(&param_alloc);
    alloc_free(&act_g);
    alloc_free(&acts);
    alloc_free(&grads);
    return 0;
}

int main(int argc, char** argv) {
    const int warmup = 50;
    const int iters = 200;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--float") == 0 || strcmp(argv[i], "--fp32") == 0 || strcmp(argv[i], "--bf16") == 0
            || strcmp(argv[i], "--half") == 0) {
            continue;
        }
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s  (precision: compile with tests/bench_gemm_conv_end2end.sh --float|--bf16)\n", argv[0]);
            return 0;
        }
        fprintf(stderr, "Unknown arg: %s\n", argv[i]);
        return 1;
    }

#ifdef PRECISION_FLOAT
    printf("bench_gemm_conv_end2end  precision=fp32  warmup=%d iters=%d\n\n", warmup, iters);
#else
    printf("bench_gemm_conv_end2end  precision=bf16  warmup=%d iters=%d\n\n", warmup, iters);
#endif

    if (run_layer(1, warmup, iters)) return 1;
    if (run_layer(2, warmup, iters)) return 1;
    return 0;
}
