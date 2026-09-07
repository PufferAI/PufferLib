// MIOpen Conv2d: forward/backward with fused bias+activation.
// AMD counterpart of src/cudnn_conv2d.cu (which keeps the NVIDIA/cuDNN
// implementation unchanged). Same struct and function interface, so
// ocean.hip.cpp can include this file where ocean.cu includes the cuDNN
// original. Included by ocean.hip.cpp (training).

#ifndef CUDNN_CONV2D_CU
#define CUDNN_CONV2D_CU

#include <hip/hip_runtime.h>
#include <miopen/miopen.h>
#include <cstdio>
#include <cassert>

#include "kernels.hip.cpp"

#ifndef CHECK_MIOPEN
#define CHECK_MIOPEN(call) do { \
    miopenStatus_t e = call; \
    if (e != miopenStatusSuccess) { \
        fprintf(stderr, "MIOpen %s:%d: %s\n", __FILE__, __LINE__, miopenGetErrorString(e)); exit(1); \
    } \
} while(0)
#endif

static inline int div_ceil(int a, int b) { return (a + b - 1) / b; }

static miopenHandle_t get_cudnn_handle() {
    static miopenHandle_t h = nullptr;
    if (!h) CHECK_MIOPEN(miopenCreate(&h));
    return h;
}

__global__ void relu_inplace_kernel(precision_t* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && data[i] < 0) data[i] = precision_t(0);
}

// ---- ConvWeights: params + batch-independent MIOpen state ----

struct ConvWeights {
    PrecisionTensor w, b;  // w: (OC, IC*K*K), b: (OC)
    int IC, OC, K, S, IH, IW, OH, OW;
    bool relu;
    miopenDataType_t dtype;
    miopenTensorDescriptor_t cudnn_bias;
    miopenTensorDescriptor_t cudnn_filt;
    miopenConvolutionDescriptor_t cudnn_conv;
    miopenActivationDescriptor_t cudnn_act;
    bool cudnn_ready;
};

// ---- ConvActivations: per-batch-size buffers + descriptors ----

struct ConvActivations {
    PrecisionTensor out, grad, saved_input;
    PrecisionTensor wgrad, bgrad;
    // Per-batch-size MIOpen state
    miopenTensorDescriptor_t cudnn_in, cudnn_out;
    miopenConvFwdAlgorithm_t fwd_algo;
    miopenConvBwdDataAlgorithm_t bwd_data_algo;
    miopenConvBwdWeightsAlgorithm_t bwd_filt_algo;
    size_t fwd_ws_bytes, bwd_data_ws_bytes, bwd_filt_ws_bytes;
    void* fwd_ws; void* bwd_data_ws; void* bwd_filt_ws;
    bool cudnn_setup;
};

static void conv_init(ConvWeights* cw, int IC, int OC, int K, int S, int IH, int IW, bool relu) {
    cw->IC = IC; cw->OC = OC; cw->K = K; cw->S = S; cw->IH = IH; cw->IW = IW;
    cw->OH = (IH - K) / S + 1; cw->OW = (IW - K) / S + 1;
    cw->relu = relu; cw->cudnn_ready = false;
}

// Create batch-independent descriptors (once)
static void conv_setup_shared(ConvWeights* cw, miopenDataType_t dt) {
    if (cw->cudnn_ready) return;
    assert(dt == miopenFloat && "MIOpen conv path supports float only");
    cw->dtype = dt;
    CHECK_MIOPEN(miopenCreateTensorDescriptor(&cw->cudnn_filt));
    CHECK_MIOPEN(miopenSet4dTensorDescriptor(cw->cudnn_filt, dt, cw->OC, cw->IC, cw->K, cw->K));
    CHECK_MIOPEN(miopenCreateConvolutionDescriptor(&cw->cudnn_conv));
    CHECK_MIOPEN(miopenInitConvolutionDescriptor(cw->cudnn_conv, miopenConvolution,
        0, 0, cw->S, cw->S, 1, 1));
    CHECK_MIOPEN(miopenCreateTensorDescriptor(&cw->cudnn_bias));
    CHECK_MIOPEN(miopenSet4dTensorDescriptor(cw->cudnn_bias, dt, 1, cw->OC, 1, 1));
    CHECK_MIOPEN(miopenCreateActivationDescriptor(&cw->cudnn_act));
    CHECK_MIOPEN(miopenSetActivationDescriptor(cw->cudnn_act,
        cw->relu ? miopenActivationRELU : miopenActivationPASTHRU, 0.0, 0.0, 0.0));
    cw->cudnn_ready = true;
}

// Setup per-activation-set MIOpen state: batch-dependent descriptors + algo search + workspace
static void conv_setup_activations(ConvWeights* cw, ConvActivations* ca, int B, miopenDataType_t dt) {
    conv_setup_shared(cw, dt);
    miopenHandle_t h = get_cudnn_handle();

    CHECK_MIOPEN(miopenCreateTensorDescriptor(&ca->cudnn_in));
    CHECK_MIOPEN(miopenSet4dTensorDescriptor(ca->cudnn_in, dt, B, cw->IC, cw->IH, cw->IW));
    CHECK_MIOPEN(miopenCreateTensorDescriptor(&ca->cudnn_out));
    CHECK_MIOPEN(miopenSet4dTensorDescriptor(ca->cudnn_out, dt, B, cw->OC, cw->OH, cw->OW));

    int returned;
    size_t ws_bytes = 0;

    miopenConvAlgoPerf_t fp;
    CHECK_MIOPEN(miopenFindConvolutionForwardAlgorithm(h,
        ca->cudnn_in, nullptr, cw->cudnn_filt, cw->w.data, cw->cudnn_conv,
        ca->cudnn_out, nullptr, 1, &returned, &fp,
        nullptr, 0, false));
    ca->fwd_algo = fp.fwd_algo;
    CHECK_MIOPEN(miopenConvolutionForwardGetWorkSpaceSize(h,
        cw->cudnn_filt, ca->cudnn_in, cw->cudnn_conv, ca->cudnn_out,
        &ws_bytes));
    ca->fwd_ws_bytes = ws_bytes;
    ca->fwd_ws = NULL; if (ca->fwd_ws_bytes > 0) hipMalloc(&ca->fwd_ws, ca->fwd_ws_bytes);

    miopenConvAlgoPerf_t ffp;
    CHECK_MIOPEN(miopenFindConvolutionBackwardWeightsAlgorithm(h,
        ca->cudnn_out, nullptr, ca->cudnn_in, nullptr, cw->cudnn_conv,
        cw->cudnn_filt, ca->wgrad.data, 1, &returned, &ffp,
        nullptr, 0, false));
    ca->bwd_filt_algo = ffp.bwd_weights_algo;
    CHECK_MIOPEN(miopenConvolutionBackwardWeightsGetWorkSpaceSize(h,
        ca->cudnn_out, ca->cudnn_in, cw->cudnn_conv, cw->cudnn_filt,
        &ws_bytes));
    ca->bwd_filt_ws_bytes = ws_bytes;
    ca->bwd_filt_ws = NULL; if (ca->bwd_filt_ws_bytes > 0) hipMalloc(&ca->bwd_filt_ws, ca->bwd_filt_ws_bytes);

    miopenConvAlgoPerf_t dp;
    CHECK_MIOPEN(miopenFindConvolutionBackwardDataAlgorithm(h,
        cw->cudnn_filt, nullptr, ca->cudnn_out, nullptr, cw->cudnn_conv,
        ca->cudnn_in, nullptr, 1, &returned, &dp,
        nullptr, 0, false));
    ca->bwd_data_algo = dp.bwd_data_algo;
    CHECK_MIOPEN(miopenConvolutionBackwardDataGetWorkSpaceSize(h,
        ca->cudnn_out, cw->cudnn_filt, cw->cudnn_conv, ca->cudnn_in,
        &ws_bytes));
    ca->bwd_data_ws_bytes = ws_bytes;
    ca->bwd_data_ws = NULL; if (ca->bwd_data_ws_bytes > 0) hipMalloc(&ca->bwd_data_ws, ca->bwd_data_ws_bytes);

    ca->cudnn_setup = true;
}

// Legacy single-setup API (for tests)
static void conv_setup(ConvWeights* cw, int B, miopenDataType_t dt) {
    (void)B;
    conv_setup_shared(cw, dt);
}

static void conv_reg_params(ConvWeights* cw, Allocator* alloc) {
    cw->w = {.shape = {cw->OC, cw->IC * cw->K * cw->K}};
    cw->b = {.shape = {cw->OC}};
    alloc_register(alloc,&cw->w); alloc_register(alloc,&cw->b);
}

static void conv_reg_train(ConvWeights* cw, ConvActivations* ca, Allocator* acts, Allocator* grads, int B, miopenDataType_t dt) {
    ca->out         = {.shape = {B * cw->OC * cw->OH * cw->OW}};
    ca->grad        = {.shape = {B * cw->OC * cw->OH * cw->OW}};
    ca->saved_input = {.shape = {B * cw->IC * cw->IH * cw->IW}};
    ca->wgrad       = {.shape = {cw->OC, cw->IC * cw->K * cw->K}};
    ca->bgrad       = {.shape = {cw->OC}};
    alloc_register(acts,&ca->out); alloc_register(acts,&ca->grad); alloc_register(acts,&ca->saved_input);
    alloc_register(grads,&ca->wgrad); alloc_register(grads,&ca->bgrad);
    conv_setup_activations(cw, ca, B, dt);
}

static void conv_reg_rollout(ConvWeights* cw, ConvActivations* ca, Allocator* alloc, int B, miopenDataType_t dt) {
    ca->out = {.shape = {B * cw->OC * cw->OH * cw->OW}};
    ca->cudnn_setup = false;
    alloc_register(alloc,&ca->out);
    conv_setup_activations(cw, ca, B, dt);
}

static void conv_init_weights(ConvWeights* cw, uint64_t* seed, hipStream_t stream) {
    PrecisionTensor wt = {.data = cw->w.data, .shape = {cw->OC, cw->IC * cw->K * cw->K}};
    puf_kaiming_init(&wt, 1.0f, (*seed)++, stream);
    hipMemsetAsync(cw->b.data, 0, numel(cw->b.shape) * sizeof(precision_t), stream);
}

// ---- Forward / Backward ----

// Fused conv + bias + activation. All NCHW. Saves input for backward.
static void conv_forward(ConvWeights* cw, ConvActivations* ca, void* input, int B, hipStream_t stream) {
    miopenHandle_t h = get_cudnn_handle();
    CHECK_MIOPEN(miopenSetStream(h, stream));
    float alpha = 1.0f, beta = 0.0f;
    if (ca->saved_input.data) {
        hipMemcpyAsync(ca->saved_input.data, input,
            (int64_t)B * cw->IC * cw->IH * cw->IW * sizeof(precision_t), hipMemcpyDeviceToDevice, stream);
    }
    CHECK_MIOPEN(miopenConvolutionForward(h,
        &alpha, ca->cudnn_in, input, cw->cudnn_filt, cw->w.data,
        cw->cudnn_conv, ca->fwd_algo,
        &beta, ca->cudnn_out, ca->out.data,
        ca->fwd_ws, ca->fwd_ws_bytes));
    CHECK_MIOPEN(miopenConvolutionForwardBias(h,
        &alpha, cw->cudnn_bias, cw->b.data,
        &beta, ca->cudnn_out, ca->out.data));
    if (cw->relu) {
        int n = B * cw->OC * cw->OH * cw->OW;
        relu_inplace_kernel<<<div_ceil(n, 256), 256, 0, stream>>>(
            (precision_t*)ca->out.data, n);
    }
}

// Backward: upstream grad in ca->grad, relu mask in ca->out.
// Caller must apply relu backward and bias grad (dtype-specific kernels).
// This does MIOpen filter grad + optional data grad.
static void conv_backward(ConvWeights* cw, ConvActivations* ca, void* input_grad, int B, hipStream_t stream) {
    (void)B;
    miopenHandle_t h = get_cudnn_handle();
    CHECK_MIOPEN(miopenSetStream(h, stream));
    float alpha = 1.0f, beta = 0.0f;

    CHECK_MIOPEN(miopenConvolutionBackwardWeights(h,
        &alpha, ca->cudnn_out, ca->grad.data, ca->cudnn_in, ca->saved_input.data,
        cw->cudnn_conv, ca->bwd_filt_algo, &beta, cw->cudnn_filt, ca->wgrad.data,
        ca->bwd_filt_ws, ca->bwd_filt_ws_bytes));

    if (input_grad) {
        CHECK_MIOPEN(miopenConvolutionBackwardData(h,
            &alpha, ca->cudnn_out, ca->grad.data, cw->cudnn_filt, cw->w.data,
            cw->cudnn_conv, ca->bwd_data_algo,
            &beta, ca->cudnn_in, input_grad,
            ca->bwd_data_ws, ca->bwd_data_ws_bytes));
    }
}

#endif // CUDNN_CONV2D_CU
