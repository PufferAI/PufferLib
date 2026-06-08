#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

namespace {

constexpr int THREADS = 256;

__global__ void point_linear_max_forward_kernel(
        const float* __restrict__ observations,
        const float* __restrict__ weight,
        const float* __restrict__ bias,
        float* __restrict__ output,
        int batch_size,
        int self_dim,
        int point_dim,
        int num_points,
        int hidden_size) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int input_dim = self_dim + num_points * point_dim;
    int point_input_dim = self_dim + point_dim;

    if (batch_idx >= batch_size) {
        return;
    }

    const float* obs_row = observations + (int64_t)batch_idx * input_dim;

    for (int hidden_idx = tid; hidden_idx < hidden_size; hidden_idx += blockDim.x) {
        const float* row = weight + (int64_t)hidden_idx * point_input_dim;
        float base = bias[hidden_idx];
        for (int d = 0; d < self_dim; ++d) {
            base += row[d] * obs_row[d];
        }

        float max_val = -FLT_MAX;
        for (int point_idx = 0; point_idx < num_points; ++point_idx) {
            const float* point = obs_row + self_dim + (int64_t)point_idx * point_dim;
            float sum = base;
            for (int d = 0; d < point_dim; ++d) {
                sum += row[self_dim + d] * point[d];
            }
            if (sum > max_val) {
                max_val = sum;
            }
        }

        output[(int64_t)batch_idx * hidden_size + hidden_idx] = max_val;
    }
}

}  // namespace

extern "C" {

int point_linear_max_forward(
        void* output,
        const void* observations,
        const void* weight,
        const void* bias,
        int batch_size,
        int self_dim,
        int point_dim,
        int num_points,
        int hidden_size) {
    dim3 block(THREADS);
    dim3 grid(batch_size);

    point_linear_max_forward_kernel<<<grid, block>>>(
        (const float*)observations,
        (const float*)weight,
        (const float*)bias,
        (float*)output,
        batch_size,
        self_dim,
        point_dim,
        num_points,
        hidden_size);

    return (int)cudaGetLastError();
}

int point_linear_max_synchronize() {
    return (int)cudaDeviceSynchronize();
}

const char* point_linear_max_error_string(int code) {
    return cudaGetErrorString((cudaError_t)code);
}

}  // extern "C"
