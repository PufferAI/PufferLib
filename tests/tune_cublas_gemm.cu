// Sweep cublasGemmEx algorithms for the same layout as puf_mm_tn (see kernels.cu).
// Default (M,N,K) matches NMMO3 conv1 ∂W GEMM at B=1024: M=OC, N=IC*K*K, K=B*OH*OW.
//
// Build: see tests/tune_cublas_gemm.sh (-DPRECISION_FLOAT => fp32; omit => bf16)

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef CUBLAS_GEMM_ALGO0
#define CUBLAS_GEMM_ALGO0 ((cublasGemmAlgo_t)0)
#endif

#ifdef PRECISION_FLOAT
typedef float precision_t;
static constexpr cudaDataType_t kCudaPrec = CUDA_R_32F;
static constexpr cublasComputeType_t kCompute = CUBLAS_COMPUTE_32F;
#else
typedef __nv_bfloat16 precision_t;
static constexpr cudaDataType_t kCudaPrec = CUDA_R_16BF;
static constexpr cublasComputeType_t kCompute = CUBLAS_COMPUTE_32F;
#endif

static void check_cuda(cudaError_t e) {
    if (e != cudaSuccess) {
        fprintf(stderr, "cuda: %s\n", cudaGetErrorString(e));
        exit(1);
    }
}

static const char* cublas_str(cublasStatus_t s) {
    switch (s) {
    case CUBLAS_STATUS_SUCCESS: return "SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "NOT_SUPPORTED";
    default: return "OTHER";
    }
}

// Same lda/ldb rules as cublasGemmExDense in kernels.cu
static inline void gemm_ex_like_puf_mm_tn(cublasHandle_t h, int M, int N, int K, const precision_t* A,
    const precision_t* B, precision_t* C, cublasGemmAlgo_t algo, cudaStream_t stream) {
    const float alpha = 1.0f, beta = 0.0f;
    cublasOperation_t op_a = CUBLAS_OP_T;
    cublasOperation_t op_b = CUBLAS_OP_N;
    int lda = (op_a == CUBLAS_OP_N) ? K : M;
    int ldb = (op_b == CUBLAS_OP_N) ? N : K;
    cublasSetStream(h, stream);
    cublasStatus_t st = cublasGemmEx(h, op_b, op_a, N, M, K, &alpha, B, kCudaPrec, ldb, A, kCudaPrec, lda,
        &beta, C, kCudaPrec, N, kCompute, algo);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasGemmEx failed: %s\n", cublas_str(st));
        exit(1);
    }
}

template<typename F>
static float time_ms(cudaStream_t stream, F fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) {
        fn(stream);
        cudaStreamSynchronize(stream);
    }
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0, stream);
    for (int i = 0; i < iters; ++i) fn(stream);
    cudaEventRecord(e1, stream);
    cudaEventSynchronize(e1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms / (float)iters;
}

static float max_abs_diff_fp32(const float* a, const float* b, int n) {
    float m = 0.f;
    for (int i = 0; i < n; ++i) m = fmaxf(m, fabsf(a[i] - b[i]));
    return m;
}

int main(int argc, char** argv) {
    int Bbatch = 1024;
    int M = 128, N = 1475, Kdim = 12288;
    int warmup = 20, iters = 100;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf(
                "Usage: %s [options]\n"
                "  Tune cublasGemmEx for the same call pattern as puf_mm_tn:\n"
                "    C = B * A with op(B)=N, op(A)=T, sizes (N,M,K) -> cublasGemmEx(..., N,M,K,...)\n"
                "Options:\n"
                "  --layer 1|2   NMMO3 conv sizes for (M,N,K) at given -B (default --layer 1)\n"
                "  -B N           batch (default 1024), used with --layer\n"
                "  -M,-N,-K       override matrix dims (after --layer, if set)\n"
                "  --warmup N     (default 20)\n"
                "  --iters N      (default 100)\n"
                "  (M,N,K) default without --layer: 128 1475 12288 (conv1 ∂W @ B=1024)\n",
                argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-B") == 0 && i + 1 < argc) Bbatch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) {
            int L = atoi(argv[++i]);
            if (L == 1) {
                const int OH = 3, OW = 4, IC = 59, OC = 128, Kk = 5;
                Kdim = Bbatch * OH * OW;
                N = IC * Kk * Kk;
                M = OC;
            } else if (L == 2) {
                const int OH = 1, OW = 2, IC = 128, OC = 128, Kk = 3;
                Kdim = Bbatch * OH * OW;
                N = IC * Kk * Kk;
                M = OC;
            } else {
                fprintf(stderr, "layer must be 1 or 2\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-M") == 0 && i + 1 < argc) M = atoi(argv[++i]);
        else if (strcmp(argv[i], "-N") == 0 && i + 1 < argc) N = atoi(argv[++i]);
        else if (strcmp(argv[i], "-K") == 0 && i + 1 < argc) Kdim = atoi(argv[++i]);
    }

    int ldc = N;
    int lenA = Kdim * M;
    int lenB = Kdim * N;
    int lenC = M * N;

    printf("tune_cublas_gemm  M=%d N=%d K=%d  (puf_mm_tn logical sizes)", M, N, Kdim);
#ifdef PRECISION_FLOAT
    printf("  dtype=fp32\n");
#else
    printf("  dtype=bf16  compute=CUBLAS_COMPUTE_32F\n");
#endif

    precision_t *dA, *dB, *dC, *dRef;
    check_cuda(cudaMalloc(&dA, (size_t)lenA * sizeof(precision_t)));
    check_cuda(cudaMalloc(&dB, (size_t)lenB * sizeof(precision_t)));
    check_cuda(cudaMalloc(&dC, (size_t)lenC * sizeof(precision_t)));
    check_cuda(cudaMalloc(&dRef, (size_t)lenC * sizeof(precision_t)));

    std::vector<float> hAf(lenA), hBf(lenB);
    unsigned seed = 12345;
    for (int i = 0; i < lenA; ++i) {
        seed = seed * 1103515245u + 12345u;
        hAf[i] = (((seed >> 16) & 0x7fff) / 16384.0f - 1.0f) * 0.25f;
    }
    for (int i = 0; i < lenB; ++i) {
        seed = seed * 1103515245u + 12345u;
        hBf[i] = (((seed >> 16) & 0x7fff) / 16384.0f - 1.0f) * 0.25f;
    }
#ifdef PRECISION_FLOAT
    check_cuda(cudaMemcpy(dA, hAf.data(), (size_t)lenA * sizeof(float), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(dB, hBf.data(), (size_t)lenB * sizeof(float), cudaMemcpyHostToDevice));
#else
    std::vector<precision_t> hA(lenA), hB(lenB);
    for (int i = 0; i < lenA; ++i) hA[i] = __float2bfloat16(hAf[i]);
    for (int i = 0; i < lenB; ++i) hB[i] = __float2bfloat16(hBf[i]);
    check_cuda(cudaMemcpy(dA, hA.data(), (size_t)lenA * sizeof(precision_t), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(dB, hB.data(), (size_t)lenB * sizeof(precision_t), cudaMemcpyHostToDevice));
#endif

    cublasHandle_t handle;
    cublasCreate(&handle);

    std::vector<float> ref_host(lenC);
    cudaStream_t stream = 0;

    auto run_ref = [&]() {
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
        gemm_ex_like_puf_mm_tn(handle, M, N, Kdim, dA, dB, dRef, CUBLAS_GEMM_DEFAULT, stream);
    };
    run_ref();
    cudaDeviceSynchronize();
#ifdef PRECISION_FLOAT
    check_cuda(cudaMemcpy(ref_host.data(), dRef, (size_t)lenC * sizeof(float), cudaMemcpyDeviceToHost));
#else
    std::vector<precision_t> ref_bf(lenC);
    check_cuda(cudaMemcpy(ref_bf.data(), dRef, (size_t)lenC * sizeof(precision_t), cudaMemcpyDeviceToHost));
    for (int i = 0; i < lenC; ++i) ref_host[i] = __bfloat162float(ref_bf[i]);
#endif

    struct Row {
        cublasGemmAlgo_t algo;
        cublasMath_t math;
        float ms;
        float max_diff;
        bool ok;
    };
    std::vector<Row> rows;

    const cublasMath_t math_modes[] = {CUBLAS_DEFAULT_MATH, CUBLAS_TENSOR_OP_MATH};
    const char* math_names[] = {"DEFAULT_MATH", "TENSOR_OP_MATH"};

    std::vector<cublasGemmAlgo_t> algos;
    algos.push_back(CUBLAS_GEMM_DEFAULT);
#ifdef CUBLAS_GEMM_DEFAULT_TENSOR_OP
    algos.push_back(CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
    for (int a = 0; a <= 23; ++a)
        algos.push_back((cublasGemmAlgo_t)((int)CUBLAS_GEMM_ALGO0 + a));

    for (size_t mi = 0; mi < sizeof(math_modes) / sizeof(math_modes[0]); ++mi) {
        cublasSetMathMode(handle, math_modes[mi]);
        for (cublasGemmAlgo_t algo : algos) {
            cublasStatus_t st = cublasSetStream(handle, stream);
            (void)st;
            const float alpha = 1.f, beta = 0.f;
            cublasOperation_t op_a = CUBLAS_OP_T;
            cublasOperation_t op_b = CUBLAS_OP_N;
            int lda = (op_a == CUBLAS_OP_N) ? Kdim : M;
            int ldb = (op_b == CUBLAS_OP_N) ? N : Kdim;
            st = cublasGemmEx(handle, op_b, op_a, N, M, Kdim, &alpha, dB, kCudaPrec, ldb, dA, kCudaPrec, lda,
                &beta, dC, kCudaPrec, N, kCompute, algo);

            if (st != CUBLAS_STATUS_SUCCESS) {
                rows.push_back({algo, math_modes[mi], 0.f, 0.f, false});
                continue;
            }
            cudaDeviceSynchronize();

            float ms = time_ms(
                stream,
                [&](cudaStream_t s) {
                    cublasGemmEx(handle, op_b, op_a, N, M, Kdim, &alpha, dB, kCudaPrec, ldb, dA, kCudaPrec, lda,
                        &beta, dC, kCudaPrec, N, kCompute, algo);
                },
                warmup, iters);

#ifdef PRECISION_FLOAT
            std::vector<float> hC(lenC);
            check_cuda(cudaMemcpy(hC.data(), dC, (size_t)lenC * sizeof(float), cudaMemcpyDeviceToHost));
            float md = max_abs_diff_fp32(ref_host.data(), hC.data(), lenC);
#else
            std::vector<precision_t> hCb(lenC);
            check_cuda(cudaMemcpy(hCb.data(), dC, (size_t)lenC * sizeof(precision_t), cudaMemcpyDeviceToHost));
            std::vector<float> hC(lenC);
            for (int i = 0; i < lenC; ++i) hC[i] = __bfloat162float(hCb[i]);
            float md = max_abs_diff_fp32(ref_host.data(), hC.data(), lenC);
#endif
            rows.push_back({algo, math_modes[mi], ms, md, true});
        }
    }

    cublasDestroy(handle);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dRef);

    float best_ms = 1e30f;
    int best_i = -1;
    for (size_t i = 0; i < rows.size(); ++i) {
        if (!rows[i].ok) continue;
        if (rows[i].ms < best_ms) {
            best_ms = rows[i].ms;
            best_i = (int)i;
        }
    }

    printf("\n%-6s %-22s %-8s %10s %12s %s\n", "algo", "math", "ok", "us/iter", "max|diff|", "note");
    printf(
        "------ ---------------------- -------- ---------- ------------ ----\n");
    for (size_t i = 0; i < rows.size(); ++i) {
        const Row& r = rows[i];
        const char* mn = (r.math == CUBLAS_DEFAULT_MATH) ? math_names[0] : math_names[1];
        if (!r.ok) {
            printf("%-6d %-22s %-8s\n", (int)r.algo, mn, "no");
            continue;
        }
        const char* tag = ((int)i == best_i) ? "best" : "";
        printf("%-6d %-22s %-8s %10.4f %12.5g %s\n", (int)r.algo, mn, "yes", r.ms * 1000.0f, r.max_diff, tag);
    }
    if (best_i >= 0) {
        printf("\nFastest OK: algo=%d math=%s  (%.4f us/iter)  max|diff|=%g vs DEFAULT/DEFAULT_MATH "
               "reference\n",
            (int)rows[best_i].algo,
            rows[best_i].math == CUBLAS_DEFAULT_MATH ? math_names[0] : math_names[1], best_ms * 1000.0f,
            rows[best_i].max_diff);
    }
    return 0;
}
