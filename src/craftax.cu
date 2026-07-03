// Craftax CUDA encoder: bag-of-embeddings, concat, projection
// Included by ocean.cu — requires precision_t, PrecisionTensor, Allocator, puf_mm, etc.

static constexpr int CX_OBS_ROWS = 9, CX_OBS_COLS = 11;
static constexpr int CX_NUM_CELLS = CX_OBS_ROWS * CX_OBS_COLS;   // 99
static constexpr int CX_CHANNELS = 8;                            // block,item,vis,5 mobs
static constexpr int CX_MAP_FLOATS = CX_NUM_CELLS * CX_CHANNELS; // 792
static constexpr int CX_NUM_SCALARS = 51;                        // inventory/status tail
static constexpr int CX_EMB_DIM = 16;
static constexpr int CX_VOCAB_TOTAL = 154;                       // sum of per-channel vocabs
static constexpr int CX_MAP_EMB = CX_NUM_CELLS * CX_EMB_DIM;     // 1584
static constexpr int CX_CONCAT = CX_MAP_EMB + CX_NUM_SCALARS;    // 1635
// Per-channel base offset into the shared embedding table (vocab caps from
// worldgen.h: block<64, item+1<8, visible<2, mob+1<16).
__constant__ int CX_OFFSETS[CX_CHANNELS] = {0, 64, 72, 74, 90, 106, 122, 138};

// Cast float accumulation buffer to precision_t.
__global__ void craftax_float_to_precision_kernel(
        precision_t* __restrict__ dst, const float* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = from_float(src[idx]);
}

// Bag-of-embeddings: sum channel embeddings per cell into concat's first
// CX_MAP_EMB columns.
__global__ void craftax_embed_bag_kernel(
        precision_t* __restrict__ concat, const precision_t* __restrict__ obs,
        const precision_t* __restrict__ embed_w, int B, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * CX_NUM_CELLS * CX_EMB_DIM) return;
    int d = idx % CX_EMB_DIM;
    int cell = (idx / CX_EMB_DIM) % CX_NUM_CELLS;
    int b = idx / (CX_NUM_CELLS * CX_EMB_DIM);
    const precision_t* ids = obs + (int64_t)b * obs_size + cell * CX_CHANNELS;
    float sum = 0.0f;
    #pragma unroll
    for (int ch = 0; ch < CX_CHANNELS; ch++) {
        int row = CX_OFFSETS[ch] + (int)to_float(ids[ch]);
        sum += to_float(embed_w[row * CX_EMB_DIM + d]);
    }
    concat[(int64_t)b * CX_CONCAT + cell * CX_EMB_DIM + d] = from_float(sum);
}

// Copy scalar tail into concat's trailing CX_NUM_SCALARS columns.
__global__ void craftax_copy_scalars_kernel(
        precision_t* __restrict__ concat, const precision_t* __restrict__ obs,
        int B, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * CX_NUM_SCALARS) return;
    int j = idx % CX_NUM_SCALARS;
    int b = idx / CX_NUM_SCALARS;
    concat[(int64_t)b * CX_CONCAT + CX_MAP_EMB + j] =
        obs[(int64_t)b * obs_size + CX_MAP_FLOATS + j];
}

// Embedding backward: scatter-add grad to every channel row that fed the sum
// (float buffer avoids bf16 atomics).
__global__ void craftax_embed_bag_backward_kernel(
        float* __restrict__ embed_wgrad_f, const precision_t* __restrict__ grad_concat,
        const precision_t* __restrict__ obs, int B, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * CX_NUM_CELLS * CX_EMB_DIM) return;
    int d = idx % CX_EMB_DIM;
    int cell = (idx / CX_EMB_DIM) % CX_NUM_CELLS;
    int b = idx / (CX_NUM_CELLS * CX_EMB_DIM);
    float g = to_float(grad_concat[(int64_t)b * CX_CONCAT + cell * CX_EMB_DIM + d]);
    const precision_t* ids = obs + (int64_t)b * obs_size + cell * CX_CHANNELS;
    #pragma unroll
    for (int ch = 0; ch < CX_CHANNELS; ch++) {
        int row = CX_OFFSETS[ch] + (int)to_float(ids[ch]);
        atomicAdd(&embed_wgrad_f[row * CX_EMB_DIM + d], g);
    }
}

struct CraftaxEncoderWeights {
    PrecisionTensor embed_w, proj_w;
    int obs_size, hidden;
};

struct CraftaxEncoderActivations {
    PrecisionTensor concat, out, saved_obs;   // concat is reused as grad_concat in backward
    PrecisionTensor embed_wgrad, proj_wgrad;
    FloatTensor embed_wgrad_f;                // float accumulation buffer for scatter-add
};

static PrecisionTensor craftax_encoder_forward(void* w, void* activations, PrecisionTensor input, cudaStream_t stream) {
    CraftaxEncoderWeights* ew = (CraftaxEncoderWeights*)w;
    CraftaxEncoderActivations* a = (CraftaxEncoderActivations*)activations;
    int B = input.shape[0];
    if (a->saved_obs.data) puf_copy(&a->saved_obs, &input, stream);

    craftax_embed_bag_kernel<<<grid_size(B * CX_NUM_CELLS * CX_EMB_DIM), BLOCK_SIZE, 0, stream>>>(
        a->concat.data, input.data, ew->embed_w.data, B, ew->obs_size);
    craftax_copy_scalars_kernel<<<grid_size(B * CX_NUM_SCALARS), BLOCK_SIZE, 0, stream>>>(
        a->concat.data, input.data, B, ew->obs_size);

    puf_mm(&a->concat, &ew->proj_w, &a->out, stream);  // (B, CONCAT) @ (hidden, CONCAT)^T
    return a->out;
}

static void craftax_encoder_backward(void* w, void* activations, PrecisionTensor grad, cudaStream_t stream) {
    CraftaxEncoderWeights* ew = (CraftaxEncoderWeights*)w;
    CraftaxEncoderActivations* a = (CraftaxEncoderActivations*)activations;
    int B = grad.shape[0];

    // proj weight grad uses forward concat values -- must run before concat is reused.
    puf_mm_tn(&grad, &a->concat, &a->proj_wgrad, stream);
    // grad w.r.t. concat = grad @ proj_w; overwrite concat in place.
    PrecisionTensor grad_concat = {.data = a->concat.data, .shape = {B, CX_CONCAT}};
    puf_mm_nn(&grad, &ew->proj_w, &grad_concat, stream);

    int embed_n = CX_VOCAB_TOTAL * CX_EMB_DIM;
    cudaMemsetAsync(a->embed_wgrad_f.data, 0, embed_n * sizeof(float), stream);
    craftax_embed_bag_backward_kernel<<<grid_size(B * CX_NUM_CELLS * CX_EMB_DIM), BLOCK_SIZE, 0, stream>>>(
        a->embed_wgrad_f.data, grad_concat.data, a->saved_obs.data, B, ew->obs_size);
    craftax_float_to_precision_kernel<<<grid_size(embed_n), BLOCK_SIZE, 0, stream>>>(
        a->embed_wgrad.data, a->embed_wgrad_f.data, embed_n);
}

static void craftax_encoder_init_weights(void* w, uint64_t* seed, cudaStream_t stream) {
    CraftaxEncoderWeights* ew = (CraftaxEncoderWeights*)w;
    puf_normal_init(&ew->embed_w, 1.0f, (*seed)++, stream);
    PrecisionTensor pw = {.data = ew->proj_w.data, .shape = {ew->hidden, CX_CONCAT}};
    puf_kaiming_init(&pw, 1.0f, (*seed)++, stream);
}

static void craftax_encoder_reg_params(void* w, Allocator* alloc) {
    CraftaxEncoderWeights* ew = (CraftaxEncoderWeights*)w;
    ew->embed_w = {.shape = {CX_VOCAB_TOTAL, CX_EMB_DIM}};
    ew->proj_w  = {.shape = {ew->hidden, CX_CONCAT}};
    alloc_register(alloc, &ew->embed_w);
    alloc_register(alloc, &ew->proj_w);
}

static void craftax_encoder_reg_train(void* w, void* activations, Allocator* acts, Allocator* grads, int B_TT) {
    CraftaxEncoderWeights* ew = (CraftaxEncoderWeights*)w;
    CraftaxEncoderActivations* a = (CraftaxEncoderActivations*)activations;
    *a = {};
    a->concat    = {.shape = {B_TT, CX_CONCAT}};
    a->out       = {.shape = {B_TT, ew->hidden}};
    a->saved_obs = {.shape = {B_TT, ew->obs_size}};
    alloc_register(acts, &a->concat); alloc_register(acts, &a->out); alloc_register(acts, &a->saved_obs);
    a->embed_wgrad   = {.shape = {CX_VOCAB_TOTAL, CX_EMB_DIM}};
    a->embed_wgrad_f = {.shape = {CX_VOCAB_TOTAL, CX_EMB_DIM}};
    a->proj_wgrad    = {.shape = {ew->hidden, CX_CONCAT}};
    alloc_register(grads, &a->embed_wgrad);
    alloc_register(acts, &a->embed_wgrad_f);
    alloc_register(grads, &a->proj_wgrad);
}

static void craftax_encoder_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    CraftaxEncoderWeights* ew = (CraftaxEncoderWeights*)w;
    CraftaxEncoderActivations* a = (CraftaxEncoderActivations*)activations;
    a->concat = {.shape = {B, CX_CONCAT}};
    a->out    = {.shape = {B, ew->hidden}};
    alloc_register(alloc, &a->concat); alloc_register(alloc, &a->out);
}

static void* craftax_encoder_create_weights(void* self) {
    Encoder* e = (Encoder*)self;
    CraftaxEncoderWeights* ew = (CraftaxEncoderWeights*)calloc(1, sizeof(CraftaxEncoderWeights));
    ew->obs_size = e->in_dim; ew->hidden = e->out_dim;
    return ew;
}
static void craftax_encoder_free_weights(void* weights) { free(weights); }
static void craftax_encoder_free_activations(void* activations) { free(activations); }
