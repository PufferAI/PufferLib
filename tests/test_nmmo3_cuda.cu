// Test harness for NMMO3 encoder — thin wrapper around ocean.cu's real implementation.
// Build: nvcc -shared -o ocean_test.so tests/test_nmmo3_cuda.cu -I pufferlib/src -lcublas -lcudnn -lcurand -Xcompiler -fPIC -O2

#define PRECISION_FLOAT
#include "../pufferlib/src/models.cu"

extern "C" {

static Encoder g_enc;
static NMMO3EncoderWeights* g_w = nullptr;
static NMMO3EncoderActivations* g_a = nullptr;
static Allocator g_param_alloc = {}, g_act_alloc = {}, g_grad_alloc = {};

void nmmo3_test_init(int B) {
    if (g_w) {
        alloc_free(&g_param_alloc);
        alloc_free(&g_act_alloc);
        alloc_free(&g_grad_alloc);
        free(g_w);
        free(g_a);
    }
    g_enc = {};
    g_enc.in_dim = 1707;
    g_enc.out_dim = 512;
    create_custom_encoder("puffer_nmmo3", &g_enc);

    g_w = (NMMO3EncoderWeights*)g_enc.create_weights(&g_enc);
    g_param_alloc = {};
    g_enc.reg_params(g_w, &g_param_alloc);
    alloc_create(&g_param_alloc);

    g_a = (NMMO3EncoderActivations*)calloc(1, sizeof(NMMO3EncoderActivations));
    g_act_alloc = {};
    g_grad_alloc = {};
    g_enc.reg_train(g_w, g_a, &g_act_alloc, &g_grad_alloc, B);
    alloc_create(&g_act_alloc);
    alloc_create(&g_grad_alloc);
}

// Copy weights from device pointers (PyTorch CUDA tensors)
void nmmo3_test_set_weights(void* c1w, void* c1b, void* c2w, void* c2b,
                             void* ew, void* pw, void* pb) {
    cudaMemcpy(g_w->conv1.w.data, c1w, numel(g_w->conv1.w.shape) * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(g_w->conv1.b.data, c1b, numel(g_w->conv1.b.shape) * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(g_w->conv2.w.data, c2w, numel(g_w->conv2.w.shape) * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(g_w->conv2.b.data, c2b, numel(g_w->conv2.b.shape) * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(g_w->embed_w.data, ew, numel(g_w->embed_w.shape) * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(g_w->proj_w.data, pw, numel(g_w->proj_w.shape) * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(g_w->proj_b.data, pb, numel(g_w->proj_b.shape) * sizeof(float), cudaMemcpyDeviceToDevice);
}

// Forward: obs and output are device float ptrs
void nmmo3_test_forward(void* output, void* obs, int B) {
    PrecisionTensor input = {.data = (precision_t*)obs, .shape = {B, 1707}};
    PrecisionTensor result = g_enc.forward(g_w, g_a, input, 0);
    cudaMemcpy(output, result.data, B * 512 * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
}

// Backward: grad is device float ptr (B, 512), modified in-place by relu backward
void nmmo3_test_backward(void* grad, int B) {
    PrecisionTensor g = {.data = (precision_t*)grad, .shape = {B, 512}};
    g_enc.backward(g_w, g_a, g, 0);
    cudaDeviceSynchronize();
}

// Extract weight gradients (device-to-device copy)
void nmmo3_test_get_conv1_wgrad(void* dst) {
    cudaMemcpy(dst, g_a->conv1.wgrad.data, numel(g_a->conv1.wgrad.shape) * sizeof(float), cudaMemcpyDeviceToDevice);
}
void nmmo3_test_get_conv1_bgrad(void* dst) {
    cudaMemcpy(dst, g_a->conv1.bgrad.data, numel(g_a->conv1.bgrad.shape) * sizeof(float), cudaMemcpyDeviceToDevice);
}
void nmmo3_test_get_conv2_wgrad(void* dst) {
    cudaMemcpy(dst, g_a->conv2.wgrad.data, numel(g_a->conv2.wgrad.shape) * sizeof(float), cudaMemcpyDeviceToDevice);
}
void nmmo3_test_get_conv2_bgrad(void* dst) {
    cudaMemcpy(dst, g_a->conv2.bgrad.data, numel(g_a->conv2.bgrad.shape) * sizeof(float), cudaMemcpyDeviceToDevice);
}
void nmmo3_test_get_proj_wgrad(void* dst) {
    cudaMemcpy(dst, g_a->proj_wgrad.data, numel(g_a->proj_wgrad.shape) * sizeof(float), cudaMemcpyDeviceToDevice);
}
void nmmo3_test_get_proj_bgrad(void* dst) {
    cudaMemcpy(dst, g_a->proj_bgrad.data, numel(g_a->proj_bgrad.shape) * sizeof(float), cudaMemcpyDeviceToDevice);
}

// Embedding weight gradient
void nmmo3_test_get_embed_wgrad(void* dst) {
    cudaMemcpy(dst, g_a->embed_wgrad.data, numel(g_a->embed_wgrad.shape) * sizeof(float), cudaMemcpyDeviceToDevice);
}

// Intermediate activations for relu mask comparison
void nmmo3_test_get_conv1_out(void* dst, int B) {
    cudaMemcpy(dst, g_a->conv1.out.data, (int64_t)B * 128 * 3 * 4 * sizeof(float), cudaMemcpyDeviceToDevice);
}

}  // extern "C"
