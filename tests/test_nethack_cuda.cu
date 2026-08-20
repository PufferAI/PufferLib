// Test harness for the Nethack encoder — thin wrapper around nethack.cu's real
// implementation. Built as a float (PRECISION_FLOAT) shared lib so finite-diff
// gradient checking is numerically meaningful.
//
// Build (from the 5c repo root; see tests/build_test.sh):
//   nvcc -shared -o tests/nethack_test.so tests/test_nethack_cuda.cu ...
#define PRECISION_FLOAT
#include <string>
#include <cstdio>
#include "../src/pufferl.cu"

extern "C" {

static Encoder g_enc;
static NethackEncoderWeights* g_w = nullptr;
static NethackEncoderActivations* g_a = nullptr;
static Allocator g_pa = {}, g_aa = {}, g_ga = {};
static Decoder g_dec;
static NethackDecoderWeights* g_dw = nullptr;
static NethackDecoderActivations* g_da = nullptr;
static Allocator g_dpa = {}, g_daa = {}, g_dga = {};
static int g_hidden = 32;

void nh_init(int B, int hidden) {
    cublas_init_handle();
    g_hidden = hidden;
    g_enc = {};
    g_enc.in_dim = NH_OBS_SIZE;
    g_enc.out_dim = hidden;
    create_nethack_encoder(&g_enc);
    g_w = (NethackEncoderWeights*)g_enc.create_weights(&g_enc);
    g_pa = {};
    g_enc.reg_params(g_w, &g_pa);
    alloc_create(&g_pa);
    g_a = (NethackEncoderActivations*)calloc(1, sizeof(NethackEncoderActivations));
    g_aa = {}; g_ga = {};
    g_enc.reg_train(g_w, g_a, &g_aa, &g_ga, B);
    alloc_create(&g_aa);
    alloc_create(&g_ga);
    uint64_t seed = 1234;
    g_enc.init_weights(g_w, &seed, 0);
    // pointer decoder, fed by g_a's inv_out (nh_enc_last set by reg_train
    // above). Its keygrad buffer feeds encoder backward — zero it so the
    // encoder-only checks stay exact until nh_dec_backward runs.
    g_dec = {};
    g_dec.hidden_dim = hidden;
    g_dec.output_dim = NH_DEC_OD;
    create_nethack_decoder(&g_dec);
    g_dw = (NethackDecoderWeights*)g_dec.create_weights(&g_dec);
    g_dpa = {};
    g_dec.reg_params(g_dw, &g_dpa);
    alloc_create(&g_dpa);
    g_da = (NethackDecoderActivations*)calloc(1, g_dec.activation_size);
    g_daa = {}; g_dga = {};
    g_dec.reg_train(g_dw, g_da, &g_daa, &g_dga, B);
    alloc_create(&g_daa);
    alloc_create(&g_dga);
    g_dec.init_weights(g_dw, &seed, 0);
    cudaMemset(g_da->keygrad.data, 0, (size_t)B * NH_INV_FLAT * sizeof(float));
    cudaDeviceSynchronize();
}

int nh_obs_size()    { return NH_OBS_SIZE; }
int nh_bl_feat()     { return NH_BL_FEAT; }
int nh_glyph_vocab() { return NH_GLYPH_VOCAB; }
int nh_embed_dim()   { return NH_EMBED_DIM; }
int nh_concat()      { return NH_CONCAT; }
int nh_grid()        { return NH_MGRID; }
int nh_dec_od()      { return NH_DEC_OD; }
int nh_dec_pad()     { return NH_DEC_PAD; }
int nh_num_actions() { return NH_ACTIONS; }
int nh_heads()       { return NH_HEADS; }

void nh_forward(void* out, void* obs, int B) {
    Prec in = {.data = (precision_t*)obs, .shape = {B, NH_OBS_SIZE}};
    Prec r = g_enc.forward(g_w, g_a, in, 0);
    cudaMemcpy(out, r.data, (size_t)B * g_hidden * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
}

void nh_backward(void* grad, int B) {
    Prec g = {.data = (precision_t*)grad, .shape = {B, g_hidden}};
    g_enc.backward(g_w, g_a, g, 0);
    cudaDeviceSynchronize();
}

// value get/set + grad get for each learnable tensor (all device float ptrs)
#define TENSOR_ACC(name, field) \
    void nh_get_##name(void* dst) { cudaMemcpy(dst, g_w->field.data, numel(g_w->field.shape) * sizeof(float), cudaMemcpyDeviceToDevice); } \
    void nh_set_##name(void* src) { cudaMemcpy(g_w->field.data, src, numel(g_w->field.shape) * sizeof(float), cudaMemcpyDeviceToDevice); cudaDeviceSynchronize(); } \
    int  nh_numel_##name()        { return (int)numel(g_w->field.shape); }
TENSOR_ACC(embed_w, embed_w)
TENSOR_ACC(ekind_w, ekind_w)
TENSOR_ACC(esub_w,  esub_w)
TENSOR_ACC(bl_w,    bl_w)
TENSOR_ACC(bl_b,    bl_b)
TENSOR_ACC(proj_w,  proj_w)
TENSOR_ACC(proj_b,  proj_b)
TENSOR_ACC(loc_w,   loc_w)
TENSOR_ACC(loc_b,   loc_b)
TENSOR_ACC(glb1_w,  glb1_w)
TENSOR_ACC(glb1_xy, glb1_xy)
TENSOR_ACC(glb1_b,  glb1_b)
TENSOR_ACC(glb2_w,  glb2_w)
TENSOR_ACC(glb2_b,  glb2_b)
TENSOR_ACC(inv1_w,  inv1_w)
TENSOR_ACC(inv1_b,  inv1_b)
TENSOR_ACC(inv1s_w, inv1s_w)
TENSOR_ACC(invt_w,  invt_w)
TENSOR_ACC(inv2_w,  inv2_w)
TENSOR_ACC(inv2_b,  inv2_b)
TENSOR_ACC(msg_w,   msg_w)
TENSOR_ACC(spk_w,   spk_w)
TENSOR_ACC(spk2_w,  spk2_w)
TENSOR_ACC(spk2_b,  spk2_b)
#if NH_ID_EMBED
TENSOR_ACC(ide_role_w, ide_role_w)
TENSOR_ACC(ide_race_w, ide_race_w)
TENSOR_ACC(ide_gend_w, ide_gend_w)
TENSOR_ACC(ide_algn_w, ide_algn_w)
#endif

#define GRAD_ACC(name, field) \
    void nh_grad_##name(void* dst) { cudaMemcpy(dst, g_a->field.data, numel(g_a->field.shape) * sizeof(float), cudaMemcpyDeviceToDevice); }
GRAD_ACC(embed_w, embed_wgrad)
GRAD_ACC(ekind_w, ekind_wgrad)
GRAD_ACC(esub_w,  esub_wgrad)
GRAD_ACC(bl_w,    bl_wgrad)
GRAD_ACC(bl_b,    bl_bgrad)
GRAD_ACC(proj_w,  proj_wgrad)
GRAD_ACC(proj_b,  proj_bgrad)
GRAD_ACC(loc_w,   loc_wgrad)
GRAD_ACC(loc_b,   loc_bgrad)
GRAD_ACC(glb1_w,  glb1_wgrad)
GRAD_ACC(glb1_xy, glb1_xygrad)
GRAD_ACC(glb1_b,  glb1_bgrad)
GRAD_ACC(glb2_w,  glb2_wgrad)
GRAD_ACC(glb2_b,  glb2_bgrad)
GRAD_ACC(inv1_w,  inv1_wgrad)
GRAD_ACC(inv1_b,  inv1_bgrad)
GRAD_ACC(inv1s_w, inv1s_wgrad)
GRAD_ACC(invt_w,  invt_wgrad)
GRAD_ACC(inv2_w,  inv2_wgrad)
GRAD_ACC(inv2_b,  inv2_bgrad)
GRAD_ACC(msg_w,   msg_wgrad)
GRAD_ACC(spk_w,   spk_wgrad)
GRAD_ACC(spk2_w,  spk2_wgrad)
GRAD_ACC(spk2_b,  spk2_bgrad)
#if NH_ID_EMBED
GRAD_ACC(ide_role_w, ide_role_wgrad)
GRAD_ACC(ide_race_w, ide_race_wgrad)
GRAD_ACC(ide_gend_w, ide_gend_wgrad)
GRAD_ACC(ide_algn_w, ide_algn_wgrad)
#endif
int nh_id_embed() { return NH_ID_EMBED; }

// ---- pointer decoder (fed by the encoder's inv_out keys) ----
// forward: encoder -> decoder directly (no mingru in the harness); the
// decoder kernels see the same activations either way.

void nh_dec_forward(void* out, void* obs, int B) {
    Prec in = {.data = (precision_t*)obs, .shape = {B, NH_OBS_SIZE}};
    Prec h = g_enc.forward(g_w, g_a, in, 0);
    Prec r = g_dec.forward(g_dw, g_da, h, 0);
    cudaMemcpy(out, r.data, (size_t)B * (NH_DEC_OD + 1) * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
}

// glogits (B, NH_DEC_OD) + gvalue (B,) device floats; grad wrt the decoder's
// hidden-state input lands in dinput (B, hidden)
void nh_dec_backward(void* glogits, void* gvalue, void* dinput, int B) {
    Float gl = {.data = (float*)glogits, .shape = {B, NH_DEC_OD}};
    Float gs = {};
    Float gv = {.data = (float*)gvalue, .shape = {B, 1}};
    Prec gi = g_dec.backward(g_dw, g_da, gl, gs, gv, 0);
    cudaMemcpy(dinput, gi.data, (size_t)B * g_hidden * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
}

// decoder fed with an EXTERNAL hidden state (e.g. the demo's mingru output);
// runs the encoder first so the decoder's inv keys (ea->inv_out) are fresh
// for the same obs.
void nh_dec_forward_hidden(void* out, void* obs, void* hidden_in, int B) {
    Prec in = {.data = (precision_t*)obs, .shape = {B, NH_OBS_SIZE}};
    g_enc.forward(g_w, g_a, in, 0);
    Prec h = {.data = (precision_t*)hidden_in, .shape = {B, g_hidden}};
    Prec r = g_dec.forward(g_dw, g_da, h, 0);
    cudaMemcpy(out, r.data, (size_t)B * (NH_DEC_OD + 1) * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
}

void nh_dec_keygrad(void* dst, int B) {
    cudaMemcpy(dst, g_da->keygrad.data, (size_t)B * NH_INV_FLAT * sizeof(float), cudaMemcpyDeviceToDevice);
}

#define DEC_ACC(name, field) \
    void nh_get_##name(void* dst) { cudaMemcpy(dst, g_dw->field.data, numel(g_dw->field.shape) * sizeof(float), cudaMemcpyDeviceToDevice); } \
    void nh_set_##name(void* src) { cudaMemcpy(g_dw->field.data, src, numel(g_dw->field.shape) * sizeof(float), cudaMemcpyDeviceToDevice); cudaDeviceSynchronize(); } \
    int  nh_numel_##name()        { return (int)numel(g_dw->field.shape); }
DEC_ACC(dec_lin_w, lin_w)
DEC_ACC(dec_q_w,   q_w)
DEC_ACC(dec_k_w,   k_w)
DEC_ACC(dec_tau,   tau)

#define DEC_GRAD(name, field) \
    void nh_grad_##name(void* dst) { cudaMemcpy(dst, g_da->field.data, numel(g_da->field.shape) * sizeof(float), cudaMemcpyDeviceToDevice); }
DEC_GRAD(dec_lin_w, lin_wgrad)
DEC_GRAD(dec_q_w,   q_wgrad)
DEC_GRAD(dec_k_w,   k_wgrad)
DEC_GRAD(dec_tau,   tau_grad)

}  // extern "C"
