#ifndef PUFFERLIB_TENSOR_H
#define PUFFERLIB_TENSOR_H

#include <stdint.h>

#define PUF_MAX_DIMS 8

typedef struct {
    float* data;
    int64_t shape[PUF_MAX_DIMS];
} FloatTensor;

typedef struct {
    unsigned char* data;
    int64_t shape[PUF_MAX_DIMS];
} ByteTensor;

typedef struct {
    long* data;
    int64_t shape[PUF_MAX_DIMS];
} LongTensor;

typedef struct {
    int* data;
    int64_t shape[PUF_MAX_DIMS];
} IntTensor;

#ifdef __CUDACC__
typedef struct {
    precision_t* data;
    int64_t shape[PUF_MAX_DIMS];
} PrecisionTensor;
#else
// C-compatible definition: precision_t is bf16 (uint16_t) or float depending on build mode.
// Only the element size matters here (used by obs_element_size() in vecenv.h).
#ifdef PRECISION_FLOAT
typedef struct {
    float* data;
    int64_t shape[PUF_MAX_DIMS];
} PrecisionTensor;
#else
typedef struct {
    uint16_t* data;
    int64_t shape[PUF_MAX_DIMS];
} PrecisionTensor;
#endif
#endif

#endif // PUFFERLIB_TENSOR_H
