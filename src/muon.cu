#include <nccl.h>

__global__ void muon_norm_reduce(float* __restrict__ out, const float* __restrict__ partials, int num_blocks) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = (tid < num_blocks) ? partials[tid] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        *out = sdata[0];
    }
}

__global__ void muon_norm_partials(float* __restrict__ partials, const precision_t* __restrict__ src, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        float v = to_float(src[i]);
        sum += v * v;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partials[blockIdx.x] = sdata[0];
    }
}

__global__ void muon_norm_apply(precision_t* __restrict__ dst, const float* __restrict__ norm_ptr, float eps, int n) {
    float inv_norm = 1.0f / fmaxf(sqrtf(*norm_ptr), eps);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float(to_float(dst[idx]) * inv_norm);
    }
}

// Nesterov with f32 momentum accumulator and precision_t gradients
__global__ void muon_nesterov(float* __restrict__ mb, precision_t* __restrict__ gc, float mu, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float m = mu * mb[idx] + to_float(gc[idx]);
        mb[idx] = m;
        gc[idx] = from_float(to_float(gc[idx]) + mu * m);
    }
}

__global__ void aurora_nesterov(float* __restrict__ mb, precision_t* __restrict__ gc,
        float mu, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = to_float(gc[idx]);
        float m = mu * mb[idx] + (1.0f - mu) * g;
        mb[idx] = m;
        gc[idx] = from_float((1.0f - mu) * g + mu * m);
    }
}

// Fused weight update: wb = wb * (1 - lr*wd) - lr * scale * update
__global__ void muon_weight_update(float* __restrict__ wb, const precision_t* __restrict__ update,
        const float* __restrict__ lr_ptr, float wd, float scale, int n) {
    float lr = *lr_ptr;
    float wd_scale = 1.0f - lr * wd;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        wb[idx] = wb[idx] * wd_scale - lr * scale * to_float(update[idx]);
    }
}

__global__ void muon_clip_norm(precision_t* __restrict__ dst,
        const float* __restrict__ sum_sq_ptr, float max_norm, float eps, int n) {
    float clip_coef = fminf(max_norm / (sqrtf(*sum_sq_ptr) + eps), 1.0f);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float(to_float(dst[idx]) * clip_coef);
    }
}

__global__ void aurora_init_row_scales(float* __restrict__ row_scale,
        const precision_t* __restrict__ src, int rows, int cols,
        bool transposed, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        int idx = transposed ? col * rows + row : row * cols + col;
        float v = to_float(src[idx]);
        sum += v * v;
    }
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        row_scale[row] = 1.0f / fmaxf(sqrtf(sdata[0]), eps);
    }
}

__global__ void aurora_apply_row_scales(precision_t* __restrict__ dst,
        const precision_t* __restrict__ src, const float* __restrict__ row_scale,
        int R, int C, bool wide) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = R * C;
    if (idx >= n) return;
    int r = idx / C;
    int c = idx - r * C;
    float scale = wide ? row_scale[c] : row_scale[r];
    dst[idx] = from_float(to_float(src[idx]) * scale);
}

__global__ void aurora_update_row_scales(float* __restrict__ row_scale,
        const precision_t* __restrict__ update, int rows, int cols,
        bool transposed, float target_row_sq, float beta, float eps_sq) {
    int row = blockIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        int idx = transposed ? col * rows + row : row * cols + col;
        float v = to_float(update[idx]);
        sum += v * v;
    }
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        float row_sq = fmaxf(sdata[0], eps_sq);
        row_scale[row] *= powf(target_row_sq / row_sq, beta);
    }
}

static constexpr double ns_coeffs[5][3] = {
    {4.0848, -6.8946, 2.9270},
    {3.9505, -6.3029, 2.6377},
    {3.7418, -5.5913, 2.3037},
    {2.8769, -3.1427, 1.2046},
    {2.8366, -3.0525, 1.2012},
};

static constexpr int muon_polar_iters = 5;
static constexpr int aurora_polar_iters = 12;
static constexpr double aurora_simple_quintic[3] = {2.0, -1.5, 0.5};

static bool aurora_matrix_eligible(const AllocEntry& e) {
    if (ndim(e.shape) < 2) return false;
    long R = e.shape[0], C = numel(e.shape) / R;
    return min(R, C) >= 2;
}

struct Muon {
    double momentum, weight_decay, aurora_weight_decay, eps;
    bool aurora;
    float lr_val_init;
    float* lr_ptr;
    float* lr_derived_ptr;
    float* norm_ptr;
    float* grad_norm_ptr;
    FloatTensor lr_puf, lr_derived_puf, ns_norm_puf, grad_norm_puf;
    FloatTensor mb_puf, aurora_row_scale;
    PrecisionTensor gram, gram_buf, x_buf, orig_buf;
    FloatTensor norm_partials;
    long max_M, max_N;
    Allocator* param_alloc;  // params allocator — shapes used by muon_step
    ncclComm_t nccl_comm;
    int world_size;
};

void muon_init(Muon* m, Allocator* param_alloc, double lr_val,
        double momentum, double eps, double weight_decay, double aurora_weight_decay,
        Allocator* alloc, bool aurora = false) {
    m->momentum = momentum;
    m->weight_decay = weight_decay;
    m->aurora_weight_decay = aurora_weight_decay;
    m->eps = eps;
    m->aurora = aurora;
    m->lr_val_init = (float)lr_val;
    m->lr_ptr = nullptr;
    m->lr_derived_ptr = nullptr;
    m->param_alloc = param_alloc;
    m->nccl_comm = nullptr;
    m->world_size = 1;
    m->max_M = 0; m->max_N = 0;
    long n = param_alloc->total_elems;
    m->lr_puf =         {.shape = {1}};
    m->lr_derived_puf = {.shape = {2}};
    m->mb_puf =         {.shape = {n}};
    m->norm_partials =  {.shape = {256}};
    m->grad_norm_puf =  {.shape = {1}};
    alloc_register(alloc, &m->lr_puf);
    alloc_register(alloc, &m->lr_derived_puf);
    alloc_register(alloc, &m->mb_puf);
    alloc_register(alloc, &m->norm_partials);
    alloc_register(alloc, &m->grad_norm_puf);
    long max_M = 0, max_N = 0, max_aurora_N = 0;
    for (int _i = 0; _i < param_alloc->num_regs; _i++) {
        AllocEntry& e = param_alloc->regs[_i];
        if (ndim(e.shape) >= 2) {
            long R = e.shape[0], C = numel(e.shape) / R;
            max_M = max(max_M, min(R, C));
            max_N = max(max_N, max(R, C));
            if (aurora && aurora_matrix_eligible(e) && R != C) {
                max_aurora_N = max(max_aurora_N, max(R, C));
            }
        }
    }
    if (max_M > 0) {
        m->max_M = max_M; m->max_N = max_N;
        m->gram =        {.shape = {max_M, max_M}};
        m->gram_buf =    {.shape = {max_M, max_M}};
        m->x_buf =       {.shape = {max_M, max_N}};
        m->ns_norm_puf = {.shape = {1}};
        alloc_register(alloc, &m->gram);
        alloc_register(alloc, &m->gram_buf);
        alloc_register(alloc, &m->x_buf);
        if (max_aurora_N > 0) {
            m->orig_buf = {.shape = {max_M, max_N}};
            m->aurora_row_scale = {.shape = {max_aurora_N}};
            alloc_register(alloc, &m->orig_buf);
            alloc_register(alloc, &m->aurora_row_scale);
        }
        alloc_register(alloc, &m->ns_norm_puf);
    }
}

void muon_post_create(Muon* m) {
    m->lr_ptr = m->lr_puf.data;
    m->lr_derived_ptr = m->lr_derived_puf.data;
    m->grad_norm_ptr = m->grad_norm_puf.data;
    if (m->ns_norm_puf.data) m->norm_ptr = m->ns_norm_puf.data;
    cudaMemcpy(m->lr_ptr, &m->lr_val_init, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(m->lr_derived_ptr, 0, 2 * sizeof(float));
    cudaMemset(m->mb_puf.data, 0, numel(m->mb_puf.shape) * sizeof(float));
}

PrecisionTensor muon_polar_project(Muon* m, PrecisionTensor x, PrecisionTensor x_buf,
        PrecisionTensor gram, PrecisionTensor gram_buf, bool tall,
        long R, long C, long M, long N, cudaStream_t stream) {
    int nblk = min((int)grid_size(numel(x.shape)), 256);
    muon_norm_partials<<<nblk, 256, 0, stream>>>(
        m->norm_partials.data, x.data, numel(x.shape));
    muon_norm_reduce<<<1, 256, 0, stream>>>(m->norm_ptr, m->norm_partials.data, nblk);
    muon_norm_apply<<<grid_size(numel(x.shape)), BLOCK_SIZE, 0, stream>>>(
        x.data, m->norm_ptr, 1e-7f, numel(x.shape));

    cublasOperation_t gram_op_a = tall ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t gram_op_b = tall ? CUBLAS_OP_N : CUBLAS_OP_T;
    for (int i = 0; i < muon_polar_iters; ++i) {
        PrecisionTensor& src = (i % 2 == 0) ? x : x_buf;
        PrecisionTensor& dst = (i % 2 == 0) ? x_buf : x;
        cublasGemmExDense(gram_op_a, gram_op_b, (int)M, (int)M, (int)N,
            src.data, src.data, gram.data, stream);
        puf_copy(&gram_buf, &gram, stream);
        puf_addmm_nn(&gram, &gram, &gram_buf, ns_coeffs[i][2], ns_coeffs[i][1], stream);
        puf_copy(&dst, &src, stream);
        cublasGemmExDense(CUBLAS_OP_N, CUBLAS_OP_N, (int)R, (int)C, (int)M,
            tall ? src.data : gram_buf.data, tall ? gram_buf.data : src.data, dst.data,
            stream, 1.0f, ns_coeffs[i][0]);
    }

    return (muon_polar_iters % 2 == 0) ? x : x_buf;
}

PrecisionTensor aurora_polar_project(Muon* m, PrecisionTensor x, PrecisionTensor x_buf,
        PrecisionTensor gram, PrecisionTensor gram_buf, bool tall,
        long R, long C, long M, long N, cudaStream_t stream) {
    int nblk = min((int)grid_size(numel(x.shape)), 256);
    muon_norm_partials<<<nblk, 256, 0, stream>>>(
        m->norm_partials.data, x.data, numel(x.shape));
    muon_norm_reduce<<<1, 256, 0, stream>>>(m->norm_ptr, m->norm_partials.data, nblk);
    muon_norm_apply<<<grid_size(numel(x.shape)), BLOCK_SIZE, 0, stream>>>(
        x.data, m->norm_ptr, 1e-7f, numel(x.shape));

    cublasOperation_t gram_op_a = tall ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t gram_op_b = tall ? CUBLAS_OP_N : CUBLAS_OP_T;
    for (int i = 0; i < aurora_polar_iters; ++i) {
        PrecisionTensor& src = (i % 2 == 0) ? x : x_buf;
        PrecisionTensor& dst = (i % 2 == 0) ? x_buf : x;
        cublasGemmExDense(gram_op_a, gram_op_b, (int)M, (int)M, (int)N,
            src.data, src.data, gram.data, stream);
        puf_copy(&gram_buf, &gram, stream);
        puf_addmm_nn(&gram, &gram, &gram_buf, aurora_simple_quintic[2],
            aurora_simple_quintic[1], stream);
        puf_copy(&dst, &src, stream);
        cublasGemmExDense(CUBLAS_OP_N, CUBLAS_OP_N, (int)R, (int)C, (int)M,
            tall ? src.data : gram_buf.data, tall ? gram_buf.data : src.data, dst.data,
            stream, 1.0f, aurora_simple_quintic[0]);
    }

    return (aurora_polar_iters % 2 == 0) ? x : x_buf;
}

void muon_step(Muon* m, FloatTensor weights, PrecisionTensor grads, float max_grad_norm, cudaStream_t stream = 0) {
    // Multi-GPU support: simple all-reduce over a contiguous grad buffer
    if (m->nccl_comm != nullptr && m->world_size > 1) {
        ncclAllReduce(grads.data, grads.data, numel(grads.shape),
            NCCL_PRECISION, ncclAvg, m->nccl_comm, stream);
    }

    // Clip gradients by norm
    int clip_blocks = min((int)grid_size(numel(grads.shape)), 256);
    muon_norm_partials<<<clip_blocks, 256, 0, stream>>>(
        m->norm_partials.data, grads.data, numel(grads.shape));
    muon_norm_reduce<<<1, 256, 0, stream>>>(m->grad_norm_ptr, m->norm_partials.data, clip_blocks);
    muon_clip_norm<<<grid_size(numel(grads.shape)), BLOCK_SIZE, 0, stream>>>(
        grads.data, m->grad_norm_ptr, max_grad_norm, 1e-6f, numel(grads.shape));

    if (!m->aurora) {
        muon_nesterov<<<grid_size(numel(m->mb_puf.shape)), BLOCK_SIZE, 0, stream>>>(
            m->mb_puf.data, grads.data, (float)m->momentum, numel(m->mb_puf.shape));
    }

    long offset = 0;
    for (int _i = 0; _i < m->param_alloc->num_regs; _i++) {
        AllocEntry& e = m->param_alloc->regs[_i];
        precision_t* gc_ptr = grads.data + offset;
        float* wb_ptr = weights.data + offset;
        long ne = numel(e.shape);
        const precision_t* update_ptr = gc_ptr;
        bool use_aurora_matrix_update = m->aurora && aurora_matrix_eligible(e);
        float param_weight_decay = use_aurora_matrix_update
            ? (float)m->aurora_weight_decay
            : (float)m->weight_decay;
        float scale = 1.0f;

        if (m->aurora) {
            if (use_aurora_matrix_update) {
                aurora_nesterov<<<grid_size(ne), BLOCK_SIZE, 0, stream>>>(
                    m->mb_puf.data + offset, gc_ptr, (float)m->momentum, ne);
            } else {
                muon_nesterov<<<grid_size(ne), BLOCK_SIZE, 0, stream>>>(
                    m->mb_puf.data + offset, gc_ptr, (float)m->momentum, ne);
            }
        }

        if (ndim(e.shape) >= 2) {
            long R = e.shape[0], C = ne / R;
            long M = min(R, C), N = max(R, C);
            bool tall = R > C;
            bool wide = R < C;
            bool use_aurora_preconditioner = use_aurora_matrix_update && R != C;
            PrecisionTensor x = {.data = gc_ptr, .shape = {R, C}};
            PrecisionTensor x_buf = {.data = m->x_buf.data, .shape = {R, C}};
            PrecisionTensor gram = {.data = m->gram.data, .shape = {M, M}};
            PrecisionTensor gram_buf = {.data = m->gram_buf.data, .shape = {M, M}};

            if (use_aurora_preconditioner) {
                PrecisionTensor orig_buf = {.data = m->orig_buf.data, .shape = {R, C}};
                puf_copy(&orig_buf, &x, stream);
                aurora_init_row_scales<<<(int)N, 256, 0, stream>>>(
                    m->aurora_row_scale.data, orig_buf.data, (int)N, (int)M, wide, 1e-7f);
                aurora_apply_row_scales<<<grid_size(ne), BLOCK_SIZE, 0, stream>>>(
                    x.data, orig_buf.data, m->aurora_row_scale.data, (int)R, (int)C, wide);
                PrecisionTensor aurora_update = aurora_polar_project(
                    m, x, x_buf, gram, gram_buf, tall, R, C, M, N, stream);
                float target_row_sq = (float)M / (float)N;
                aurora_update_row_scales<<<(int)N, 256, 0, stream>>>(
                    m->aurora_row_scale.data, aurora_update.data, (int)N, (int)M, wide,
                    target_row_sq, 0.5f, 1e-14f);
                aurora_apply_row_scales<<<grid_size(ne), BLOCK_SIZE, 0, stream>>>(
                    x.data, orig_buf.data, m->aurora_row_scale.data, (int)R, (int)C, wide);
            }

            if (use_aurora_preconditioner) {
                PrecisionTensor aurora_update = aurora_polar_project(
                    m, x, x_buf, gram, gram_buf, tall, R, C, M, N, stream);
                update_ptr = aurora_update.data;
            } else {
                PrecisionTensor muon_update = muon_polar_project(
                    m, x, x_buf, gram, gram_buf, tall, R, C, M, N, stream);
                update_ptr = muon_update.data;
            }
            scale = sqrtf(fmaxf(1.0f, (float)R / (float)C));
        }

        muon_weight_update<<<grid_size(ne), BLOCK_SIZE, 0, stream>>>(
            wb_ptr, update_ptr, m->lr_ptr, param_weight_decay, scale, (int)ne);
        offset += ne;
    }
}
