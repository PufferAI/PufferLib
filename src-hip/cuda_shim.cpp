// cuda->HIP forwarding shims for the C env libraries on AMD.
#include <hip/hip_runtime.h>
typedef int cudaError_t;
extern "C" {
cudaError_t cudaHostAlloc(void** p, size_t s, unsigned int f) { return hipHostAlloc(p, s, f); }
cudaError_t cudaMalloc(void** p, size_t s) { return hipMalloc(p, s); }
cudaError_t cudaMemcpy(void* d, const void* s, size_t n, int k) { return hipMemcpy(d, s, n, (hipMemcpyKind)k); }
cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, int k, void* st) { return hipMemcpyAsync(d, s, n, (hipMemcpyKind)k, (hipStream_t)st); }
cudaError_t cudaMemset(void* d, int v, size_t n) { return hipMemset(d, v, n); }
cudaError_t cudaFree(void* p) { return hipFree(p); }
cudaError_t cudaFreeHost(void* p) { return hipHostFree(p); }
cudaError_t cudaSetDevice(int i) { return hipSetDevice(i); }
cudaError_t cudaDeviceSynchronize(void) { return hipDeviceSynchronize(); }
cudaError_t cudaStreamSynchronize(void* s) { return hipStreamSynchronize((hipStream_t)s); }
cudaError_t cudaStreamCreateWithFlags(void** s, unsigned int f) { return hipStreamCreateWithFlags((hipStream_t*)s, f); }
cudaError_t cudaStreamQuery(void* s) { return hipStreamQuery((hipStream_t)s); }
const char* cudaGetErrorString(cudaError_t e) { return hipGetErrorString((hipError_t)e); }
}
