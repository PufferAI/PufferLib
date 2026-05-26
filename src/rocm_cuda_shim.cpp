#include <hip/hip_runtime.h>

extern "C" {

hipError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags) {
    return hipHostMalloc(ptr, size, flags);
}

hipError_t cudaMalloc(void** ptr, size_t size) {
    return hipMalloc(ptr, size);
}

hipError_t cudaMemcpy(void* dst, const void* src, size_t size, int kind) {
    return hipMemcpy(dst, src, size, (hipMemcpyKind)kind);
}

hipError_t cudaMemcpyAsync(void* dst, const void* src, size_t size, int kind, hipStream_t stream) {
    return hipMemcpyAsync(dst, src, size, (hipMemcpyKind)kind, stream);
}

hipError_t cudaMemset(void* dst, int value, size_t size) {
    return hipMemset(dst, value, size);
}

hipError_t cudaFree(void* ptr) {
    return hipFree(ptr);
}

hipError_t cudaFreeHost(void* ptr) {
    return hipHostFree(ptr);
}

hipError_t cudaSetDevice(int device) {
    return hipSetDevice(device);
}

hipError_t cudaDeviceSynchronize(void) {
    return hipDeviceSynchronize();
}

hipError_t cudaStreamSynchronize(hipStream_t stream) {
    return hipStreamSynchronize(stream);
}

hipError_t cudaStreamCreateWithFlags(hipStream_t* stream, unsigned int flags) {
    return hipStreamCreateWithFlags(stream, flags);
}

hipError_t cudaStreamQuery(hipStream_t stream) {
    return hipStreamQuery(stream);
}

const char* cudaGetErrorString(hipError_t error) {
    return hipGetErrorString(error);
}

}
