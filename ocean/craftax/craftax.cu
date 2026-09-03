// Craftax CUDA encoder: one-hot-by-category bag-of-embeddings, concat, projection.
// Packed obs is 9x11 cells x 8 integer channels (block, item, visible, 5 mobs)
// plus a 51-float inventory/status tail. Included by src/ocean.cu.

static constexpr int CX_OBS_ROWS = 9, CX_OBS_COLS = 11;
static constexpr int CX_NUM_CELLS = CX_OBS_ROWS * CX_OBS_COLS;   // 99
static constexpr int CX_CHANNELS = 8;                            // block,item,vis,5 mobs
static constexpr int CX_MAP_FLOATS = CX_NUM_CELLS * CX_CHANNELS; // 792
static constexpr int CX_NUM_SCALARS = 51;                        // inventory/status tail
static constexpr int CX_EMB_DIM = 16;
static constexpr int CX_VOCAB_TOTAL = 154;                       // sum of per-channel vocabs
static constexpr int CX_MAP_EMB = CX_NUM_CELLS * CX_EMB_DIM;     // 1584
static constexpr int CX_CONCAT = CX_MAP_EMB + CX_NUM_SCALARS;    // 1635
static constexpr int CX_EMBED_N = CX_VOCAB_TOTAL * CX_EMB_DIM;   // 2464
static_assert(CX_MAP_FLOATS + CX_NUM_SCALARS == OBS_SIZE,
    "craftax encoder expects packed 9x11x8+51 observations");
// Per-channel base offset into the shared embedding table (vocab caps:
// block<64, item+1<8, visible<2, mob+1<16).
__constant__ int CX_OFFSETS[CX_CHANNELS] = {0, 64, 72, 74, 90, 106, 122, 138};
// 2^24 fixed-point: integer atomics are associative (nmmo3 embed scatter).
#define CX_FXP 16777216.0f

__global__ void craftax_fxp_to_precision_kernel(
        precision_t* __restrict__ dst, const long long* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float((float)((double)src[idx] * (1.0 / (double)CX_FXP)));
    }
}

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

// Per-block int64 histogram over the 154x16 table, then one global integer
// atomic per table entry. Integer add is associative so scatter order does
// not change bits.
__global__ void craftax_embed_bag_backward_kernel(
        long long* __restrict__ embed_wgrad_i,
        const precision_t* __restrict__ grad_concat,
        const precision_t* __restrict__ obs, int B, int obs_size) {
    __shared__ long long acc[CX_EMBED_N];
    for (int i = threadIdx.x; i < CX_EMBED_N; i += blockDim.x) {
        acc[i] = 0;
    }
    __syncthreads();
    int total = B * CX_NUM_CELLS * CX_EMB_DIM;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
            idx += blockDim.x * gridDim.x) {
        int d = idx % CX_EMB_DIM;
        int cell = (idx / CX_EMB_DIM) % CX_NUM_CELLS;
        int b = idx / (CX_NUM_CELLS * CX_EMB_DIM);
        const precision_t* ids = obs + (int64_t)b * obs_size + cell * CX_CHANNELS;
        long long g = __float2ll_rn(to_float(grad_concat[
            (int64_t)b * CX_CONCAT + cell * CX_EMB_DIM + d]) * CX_FXP);
        #pragma unroll
        for (int ch = 0; ch < CX_CHANNELS; ch++) {
            int row = CX_OFFSETS[ch] + (int)to_float(ids[ch]);
            atomicAdd((unsigned long long*)&acc[row * CX_EMB_DIM + d],
                (unsigned long long)g);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < CX_EMBED_N; i += blockDim.x) {
        if (acc[i] != 0) {
            atomicAdd((unsigned long long*)&embed_wgrad_i[i],
                (unsigned long long)acc[i]);
        }
    }
}

struct CraftaxEncoderWeights {
    Prec embed_w, proj_w;
    int obs_size, hidden;
};

struct CraftaxEncoderActivations {
    Prec concat, out, saved_obs;
    Prec embed_wgrad, proj_wgrad;
    Long embed_wgrad_i;
};

static Prec craftax_encoder_forward(
        void* w, void* activations, Prec input, cudaStream_t stream) {
    CraftaxEncoderWeights* ew = (CraftaxEncoderWeights*)w;
    CraftaxEncoderActivations* a = (CraftaxEncoderActivations*)activations;
    int B = input.shape[0];
    if (a->saved_obs.data) puf_copy(&a->saved_obs, &input, stream);

    craftax_embed_bag_kernel<<<grid_size(B * CX_NUM_CELLS * CX_EMB_DIM), BLOCK_SIZE, 0, stream>>>(
        a->concat.data, input.data, ew->embed_w.data, B, ew->obs_size);
    craftax_copy_scalars_kernel<<<grid_size(B * CX_NUM_SCALARS), BLOCK_SIZE, 0, stream>>>(
        a->concat.data, input.data, B, ew->obs_size);

    puf_mm(&a->concat, &ew->proj_w, &a->out, stream);
    return a->out;
}

static void craftax_encoder_backward(
        void* w, void* activations, Prec grad, cudaStream_t stream) {
    CraftaxEncoderWeights* ew = (CraftaxEncoderWeights*)w;
    CraftaxEncoderActivations* a = (CraftaxEncoderActivations*)activations;
    int B = grad.shape[0];

    puf_mm_tn(&grad, &a->concat, &a->proj_wgrad, stream);
    Prec grad_concat = {.data = a->concat.data, .shape = {B, CX_CONCAT}};
    puf_mm_nn(&grad, &ew->proj_w, &grad_concat, stream);

    cudaMemsetAsync(a->embed_wgrad_i.data, 0, CX_EMBED_N * sizeof(long), stream);
    int blocks = (B * CX_NUM_CELLS * CX_EMB_DIM) / BLOCK_SIZE;
    if (blocks < 1) {
        blocks = 1;
    }
    if (blocks > 2048) {
        blocks = 2048;
    }
    craftax_embed_bag_backward_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        (long long*)a->embed_wgrad_i.data, grad_concat.data, a->saved_obs.data,
        B, ew->obs_size);
    craftax_fxp_to_precision_kernel<<<grid_size(CX_EMBED_N), BLOCK_SIZE, 0, stream>>>(
        a->embed_wgrad.data, (long long*)a->embed_wgrad_i.data, CX_EMBED_N);
}

static void craftax_encoder_init_weights(void* w, ulong* seed, cudaStream_t stream) {
    CraftaxEncoderWeights* ew = (CraftaxEncoderWeights*)w;
    puf_normal_init(&ew->embed_w, 1.0f, (*seed)++, stream);
    Prec pw = {.data = ew->proj_w.data, .shape = {ew->hidden, CX_CONCAT}};
    puf_kaiming_init(&pw, 1.0f, (*seed)++, stream);
}

static void craftax_encoder_reg_params(void* w, Allocator* alloc) {
    CraftaxEncoderWeights* ew = (CraftaxEncoderWeights*)w;
    ew->embed_w = {.shape = {CX_VOCAB_TOTAL, CX_EMB_DIM}};
    ew->proj_w  = {.shape = {ew->hidden, CX_CONCAT}};
    alloc_register(alloc, &ew->embed_w);
    alloc_register(alloc, &ew->proj_w);
}

static void craftax_encoder_reg_train(
        void* w, void* activations, Allocator* acts, Allocator* grads, int B_TT) {
    CraftaxEncoderWeights* ew = (CraftaxEncoderWeights*)w;
    CraftaxEncoderActivations* a = (CraftaxEncoderActivations*)activations;
    *a = {};
    a->concat    = {.shape = {B_TT, CX_CONCAT}};
    a->out       = {.shape = {B_TT, ew->hidden}};
    a->saved_obs = {.shape = {B_TT, ew->obs_size}};
    alloc_register(acts, &a->concat);
    alloc_register(acts, &a->out);
    alloc_register(acts, &a->saved_obs);
    a->embed_wgrad   = {.shape = {CX_VOCAB_TOTAL, CX_EMB_DIM}};
    a->embed_wgrad_i = {.shape = {CX_VOCAB_TOTAL, CX_EMB_DIM}};
    a->proj_wgrad    = {.shape = {ew->hidden, CX_CONCAT}};
    alloc_register(grads, &a->embed_wgrad);
    alloc_register(acts, &a->embed_wgrad_i);
    alloc_register(grads, &a->proj_wgrad);
}

static void craftax_encoder_reg_rollout(
        void* w, void* activations, Allocator* alloc, int B) {
    CraftaxEncoderWeights* ew = (CraftaxEncoderWeights*)w;
    CraftaxEncoderActivations* a = (CraftaxEncoderActivations*)activations;
    *a = {};
    a->concat = {.shape = {B, CX_CONCAT}};
    a->out    = {.shape = {B, ew->hidden}};
    alloc_register(alloc, &a->concat);
    alloc_register(alloc, &a->out);
}

static void* craftax_encoder_create_weights(void* self) {
    Encoder* e = (Encoder*)self;
    CraftaxEncoderWeights* ew =
        (CraftaxEncoderWeights*)calloc(1, sizeof(CraftaxEncoderWeights));
    ew->obs_size = e->in_dim;
    ew->hidden = e->out_dim;
    return ew;
}

static void create_craftax_encoder(Encoder* enc) {
    fprintf(stderr,
        "craftax: using category embedding encoder (obs=%d hidden=%d concat=%d)\n",
        enc->in_dim, enc->out_dim, CX_CONCAT);
    *enc = Encoder{
        .forward = craftax_encoder_forward,
        .backward = craftax_encoder_backward,
        .init_weights = craftax_encoder_init_weights,
        .reg_params = craftax_encoder_reg_params,
        .reg_train = craftax_encoder_reg_train,
        .reg_rollout = craftax_encoder_reg_rollout,
        .create_weights = craftax_encoder_create_weights,
        .in_dim = enc->in_dim, .out_dim = enc->out_dim,
        .activation_size = sizeof(CraftaxEncoderActivations),
    };
}
