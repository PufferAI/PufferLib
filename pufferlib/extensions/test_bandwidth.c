// test_bandwidth.c
// Minimal CUDA bandwidth test using cudaMemcpyAsync
//
// Build: ./scripts/build_bandwidth.sh
// Run: ./test_bandwidth [mode] [block_size_bytes] [sync_each]
//
// Example: ./test_bandwidth h2d 1048576 0    # 1MB blocks, no sync
// Example: ./test_bandwidth h2d 4096 1       # 4KB blocks, sync after each

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>

// Forward declare CUDA types
typedef int cudaError_t;
typedef struct CUstream_st* cudaStream_t;
typedef int cudaMemcpyKind;
#define cudaSuccess 0
#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2
#define cudaHostAllocPortable 1
#define cudaStreamNonBlocking 1

extern cudaError_t cudaHostAlloc(void**, size_t, unsigned int);
extern cudaError_t cudaMalloc(void**, size_t);
extern cudaError_t cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind);
extern cudaError_t cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
extern cudaError_t cudaFree(void*);
extern cudaError_t cudaFreeHost(void*);
extern cudaError_t cudaSetDevice(int);
extern cudaError_t cudaDeviceSynchronize(void);
extern cudaError_t cudaStreamSynchronize(cudaStream_t);
extern cudaError_t cudaStreamCreateWithFlags(cudaStream_t*, unsigned int);
extern cudaError_t cudaStreamDestroy(cudaStream_t);
extern const char* cudaGetErrorString(cudaError_t);

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s: %s (error %d)\n",    \
                    #call, cudaGetErrorString(err), (int)err);      \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while(0)

const float TIMEOUT_SEC = 5.0f;

static float get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

void test_bandwidth_h2d(size_t block_size, bool sync_each, int num_streams) {
    printf("Host->Device: block_size=%zu bytes, sync_each=%d, streams=%d\n",
           block_size, sync_each, num_streams);

    CHECK_CUDA(cudaSetDevice(0));

    // Allocate pinned host memory
    void* host_buf;
    CHECK_CUDA(cudaHostAlloc(&host_buf, block_size, cudaHostAllocPortable));
    memset(host_buf, 0x42, block_size);

    // Allocate device memory
    void* dev_buf;
    CHECK_CUDA(cudaMalloc(&dev_buf, block_size));

    // Create streams
    cudaStream_t* streams = (cudaStream_t*)calloc(num_streams, sizeof(cudaStream_t));
    for (int i = 0; i < num_streams; i++) {
        CHECK_CUDA(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    }

    // Warmup
    for (int i = 0; i < 100; i++) {
        CHECK_CUDA(cudaMemcpyAsync(dev_buf, host_buf, block_size, cudaMemcpyHostToDevice, streams[0]));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed run - check timeout every 10000 iterations
    float start = get_time_sec();
    long iters = 0;
    float elapsed = 0;
    while (elapsed < TIMEOUT_SEC) {
        for (int i = 0; i < 10000; i++) {
            cudaStream_t stream = streams[(iters + i) % num_streams];
            CHECK_CUDA(cudaMemcpyAsync(dev_buf, host_buf, block_size, cudaMemcpyHostToDevice, stream));
            if (sync_each) {
                CHECK_CUDA(cudaStreamSynchronize(stream));
            }
        }
        iters += 10000;
        CHECK_CUDA(cudaDeviceSynchronize());
        elapsed = get_time_sec() - start;
    }

    double bytes = (double)block_size * iters;
    double gbps = (bytes / elapsed) / 1e9;
    printf("  iters=%ld, elapsed=%.3fs, total=%.2f GB, throughput=%.2f GB/s\n\n", iters, elapsed, bytes/1e9, gbps);

    // Cleanup
    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    free(streams);
    cudaFreeHost(host_buf);
    cudaFree(dev_buf);
}

void test_bandwidth_d2h(size_t block_size, bool sync_each, int num_streams) {
    printf("Device->Host: block_size=%zu bytes, sync_each=%d, streams=%d\n",
           block_size, sync_each, num_streams);

    CHECK_CUDA(cudaSetDevice(0));

    // Allocate pinned host memory
    void* host_buf;
    CHECK_CUDA(cudaHostAlloc(&host_buf, block_size, cudaHostAllocPortable));

    // Allocate device memory
    void* dev_buf;
    CHECK_CUDA(cudaMalloc(&dev_buf, block_size));

    // Create streams
    cudaStream_t* streams = (cudaStream_t*)calloc(num_streams, sizeof(cudaStream_t));
    for (int i = 0; i < num_streams; i++) {
        CHECK_CUDA(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    }

    // Warmup
    for (int i = 0; i < 100; i++) {
        CHECK_CUDA(cudaMemcpyAsync(host_buf, dev_buf, block_size, cudaMemcpyDeviceToHost, streams[0]));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed run
    float start = get_time_sec();
    long iters = 0;
    float elapsed = 0;
    while (elapsed < TIMEOUT_SEC) {
        for (int i = 0; i < 10000; i++) {
            cudaStream_t stream = streams[(iters + i) % num_streams];
            CHECK_CUDA(cudaMemcpyAsync(host_buf, dev_buf, block_size, cudaMemcpyDeviceToHost, stream));
            if (sync_each) {
                CHECK_CUDA(cudaStreamSynchronize(stream));
            }
        }
        iters += 10000;
        CHECK_CUDA(cudaDeviceSynchronize());
        elapsed = get_time_sec() - start;
    }

    double bytes = (double)block_size * iters;
    double gbps = (bytes / elapsed) / 1e9;
    printf("  iters=%ld, elapsed=%.3fs, total=%.2f GB, throughput=%.2f GB/s\n\n", iters, elapsed, bytes/1e9, gbps);

    // Cleanup
    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    free(streams);
    cudaFreeHost(host_buf);
    cudaFree(dev_buf);
}

void test_multi_buffer(size_t block_size, int num_buffers, bool sync_each) {
    printf("Multi-buffer H2D: block_size=%zu, buffers=%d, sync_each=%d\n",
           block_size, num_buffers, sync_each);

    CHECK_CUDA(cudaSetDevice(0));

    // Allocate multiple pinned host buffers and device buffers
    void** host_bufs = (void**)calloc(num_buffers, sizeof(void*));
    void** dev_bufs = (void**)calloc(num_buffers, sizeof(void*));
    cudaStream_t* streams = (cudaStream_t*)calloc(num_buffers, sizeof(cudaStream_t));

    for (int i = 0; i < num_buffers; i++) {
        CHECK_CUDA(cudaHostAlloc(&host_bufs[i], block_size, cudaHostAllocPortable));
        CHECK_CUDA(cudaMalloc(&dev_bufs[i], block_size));
        CHECK_CUDA(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        memset(host_bufs[i], 0x42 + i, block_size);
    }

    // Warmup
    for (int i = 0; i < 100; i++) {
        int buf = i % num_buffers;
        CHECK_CUDA(cudaMemcpyAsync(dev_bufs[buf], host_bufs[buf], block_size, cudaMemcpyHostToDevice, streams[buf]));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed run - round-robin through buffers
    float start = get_time_sec();
    long iters = 0;
    float elapsed = 0;
    while (elapsed < TIMEOUT_SEC) {
        for (int i = 0; i < 10000; i++) {
            int buf = (iters + i) % num_buffers;
            CHECK_CUDA(cudaMemcpyAsync(dev_bufs[buf], host_bufs[buf], block_size, cudaMemcpyHostToDevice, streams[buf]));
            if (sync_each) {
                CHECK_CUDA(cudaStreamSynchronize(streams[buf]));
            }
        }
        iters += 10000;
        CHECK_CUDA(cudaDeviceSynchronize());
        elapsed = get_time_sec() - start;
    }

    double bytes = (double)block_size * iters;
    double gbps = (bytes / elapsed) / 1e9;
    printf("  iters=%ld, elapsed=%.3fs, total=%.2f GB, throughput=%.2f GB/s\n\n", iters, elapsed, bytes/1e9, gbps);

    // Cleanup
    for (int i = 0; i < num_buffers; i++) {
        cudaStreamDestroy(streams[i]);
        cudaFreeHost(host_bufs[i]);
        cudaFree(dev_bufs[i]);
    }
    free(streams);
    free(host_bufs);
    free(dev_bufs);
}

void print_usage(const char* prog) {
    printf("Usage: %s [mode] [options]\n\n", prog);
    printf("Modes:\n");
    printf("  %s h2d [block_size] [sync_each] [num_streams]\n", prog);
    printf("      Host->Device transfer test\n\n");
    printf("  %s d2h [block_size] [sync_each] [num_streams]\n", prog);
    printf("      Device->Host transfer test\n\n");
    printf("  %s multi [block_size] [num_buffers] [sync_each]\n", prog);
    printf("      Multi-buffer round-robin test (each buffer has own stream)\n\n");
    printf("  %s sweep\n", prog);
    printf("      Run tests across multiple block sizes\n\n");
    printf("Defaults: block_size=1MB, sync_each=0, streams=1\n");
    printf("Timeout: %.1f seconds per test\n\n", TIMEOUT_SEC);
    printf("PCIe 3.0 x16 theoretical: ~12.5 GB/s\n");
    printf("PCIe 4.0 x16 theoretical: ~25 GB/s\n");
    printf("PCIe 5.0 x16 theoretical: ~50 GB/s\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char* mode = argv[1];

    if (strcmp(mode, "h2d") == 0) {
        size_t block_size = argc > 2 ? (size_t)atol(argv[2]) : 1048576;
        bool sync_each = argc > 3 ? atoi(argv[3]) : 0;
        int num_streams = argc > 4 ? atoi(argv[4]) : 1;
        test_bandwidth_h2d(block_size, sync_each, num_streams);
    }
    else if (strcmp(mode, "d2h") == 0) {
        size_t block_size = argc > 2 ? (size_t)atol(argv[2]) : 1048576;
        bool sync_each = argc > 3 ? atoi(argv[3]) : 0;
        int num_streams = argc > 4 ? atoi(argv[4]) : 1;
        test_bandwidth_d2h(block_size, sync_each, num_streams);
    }
    else if (strcmp(mode, "multi") == 0) {
        size_t block_size = argc > 2 ? (size_t)atol(argv[2]) : 1048576;
        int num_buffers = argc > 3 ? atoi(argv[3]) : 4;
        bool sync_each = argc > 4 ? atoi(argv[4]) : 0;
        test_multi_buffer(block_size, num_buffers, sync_each);
    }
    else if (strcmp(mode, "sweep") == 0) {
        printf("=== Block size sweep (H2D, no sync, 1 stream) ===\n\n");
        size_t sizes[] = {1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216};
        int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
        for (int i = 0; i < num_sizes; i++) {
            test_bandwidth_h2d(sizes[i], false, 1);
        }

        printf("=== Block size sweep (D2H, no sync, 1 stream) ===\n\n");
        for (int i = 0; i < num_sizes; i++) {
            test_bandwidth_d2h(sizes[i], false, 1);
        }

        printf("=== Sync comparison (1MB blocks) ===\n\n");
        test_bandwidth_h2d(1048576, false, 1);
        test_bandwidth_h2d(1048576, true, 1);

        printf("=== Multi-stream comparison (1MB blocks, no sync) ===\n\n");
        test_bandwidth_h2d(1048576, false, 1);
        test_bandwidth_h2d(1048576, false, 4);
        test_bandwidth_h2d(1048576, false, 8);
    }
    else {
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
