#ifndef PUFFERLIB_METAL_PLATFORM_H
#define PUFFERLIB_METAL_PLATFORM_H

#import <Metal/Metal.h>
#include <mach/mach_time.h>

#define ACCELERATE_NEW_LAPACK
#import <Accelerate/Accelerate.h>

#include "puf_types.h"
#include <cassert>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

struct MetalStream {
  id<MTL4CommandAllocator> allocator;
  id<MTL4CommandBuffer> cmd;
  id<MTL4ComputeCommandEncoder> enc;
  id<MTL4ArgumentTable> arg_table;
  id<MTLSharedEvent> sync_event;
  id<MTLBuffer> const_ring;
  NSUInteger const_ring_offset = 0;
  uint64_t bound_addresses[32] = {};

  bool enc_active = false;
  bool pending_work = false;
  bool flushed = false;
  uint64_t flush_event_val = 0;
  uint64_t sync_event_value = 0;

  void begin();
  void compute_encoder();
  void end_compute();
  void sync();
  void flush();
  void commit_chunk();
  void wait_completed();
};

struct WrappedBuffer {
  char *base;
  int64_t size;
  id<MTLBuffer> buffer;
};

static const NSUInteger MTL_CONST_RING_SIZE = 2 * 1024 * 1024;

struct MetalContext {
  id<MTLDevice> device;
  id<MTLLibrary> library;
  NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipelines;

  id<MTLComputePipelineState> tensor_ops_gemm_nt_f32;
  id<MTLComputePipelineState> tensor_ops_gemm_nn_f32;
  id<MTLComputePipelineState> tensor_ops_gemm_tn_f32;
  id<MTLComputePipelineState> tensor_ops_gemm_nt_f16;
  id<MTLComputePipelineState> tensor_ops_gemm_nn_f16;
  id<MTLComputePipelineState> tensor_ops_gemm_tn_f16;

  id<MTL4CommandQueue> queue;
  id<MTL4CommandQueue> train_queue;
  id<MTLResidencySet> residency_set;

  MetalStream stream;
  MetalStream train_stream;
  std::vector<WrappedBuffer> buffers;
};
void mtl_init();
MetalContext *mtl_ctx();
void *mtl_stream();
void *mtl_train_stream();
static inline MetalStream *mtl_resolve_stream(cudaStream_t s) {
  return s ? (MetalStream *)s : (MetalStream *)mtl_stream();
}

static inline void mtl_ensure_stream_synced(cudaStream_t s) {
  MetalStream *ms = mtl_resolve_stream(s);
  if (ms->flushed) {
    ms->wait_completed();
  } else if (ms->enc_active || ms->pending_work) {
    ms->sync();
  }
}

void *mtl_create_stream();
void mtl_destroy_stream(void *stream);

static inline void *mtl_alloc_scratch(int64_t bytes) {
  int64_t page = 16384;
  int64_t alloc = ((bytes + page - 1) / page) * page;
  void *ptr = nullptr;
  posix_memalign(&ptr, page, alloc);
  assert(ptr && "mtl_alloc_scratch: posix_memalign failed");
  memset(ptr, 0, alloc);
  MetalContext *ctx = mtl_ctx();
  id<MTLBuffer> buf = [ctx->device newBufferWithBytesNoCopy:ptr
                                                     length:(NSUInteger)alloc
                                                    options:MTLResourceStorageModeShared
                                                deallocator:nil];
  assert(buf);
  ctx->buffers.push_back({(char *)ptr, alloc, buf});
  [ctx->residency_set addAllocation:buf];
  [ctx->residency_set commit];
  [ctx->residency_set requestResidency];
  return ptr;
}

void mtl_destroy();
void mtl_kernels_reset();
id<MTLBuffer> mtl_wrap_allocator(Allocator *alloc);
id<MTLBuffer> mtl_buffer_for(const PufTensor &t, NSUInteger *out_offset);
id<MTLComputePipelineState> mtl_pipeline(const char *name);
inline void mtl_set_pso(MetalStream *ms, id<MTLComputePipelineState> pso) {
  [ms->enc setComputePipelineState:pso];
}

inline void mtl_set_tensor(MetalStream *ms, const PufTensor &t,
                           uint32_t index) {
  NSUInteger offset;
  id<MTLBuffer> buf = mtl_buffer_for(t, &offset);
  uint64_t addr = buf.gpuAddress + offset;
  if (ms->bound_addresses[index] != addr) {
    [ms->arg_table setAddress:addr atIndex:index];
    ms->bound_addresses[index] = addr;
  }
}

id<MTLBuffer> mtl_buffer_for_ptr(const void *ptr, NSUInteger *out_offset);

inline bool mtl_const_ring_reserve_range(NSUInteger current_offset,
                                         NSUInteger raw_size,
                                         NSUInteger *next_offset) {
  if (raw_size > MTL_CONST_RING_SIZE ||
      current_offset > MTL_CONST_RING_SIZE) {
    return false;
  }

  NSUInteger aligned = (raw_size + 15) & ~(NSUInteger)15;
  if (aligned > MTL_CONST_RING_SIZE - current_offset) {
    return false;
  }

  *next_offset = current_offset + aligned;
  return true;
}

inline int mtl_parse_int_config_value(const char *key, double value) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument(std::string(key) + " must be a finite integer");
  }
  double rounded = std::round(value);
  if (rounded < (double)std::numeric_limits<int>::min() ||
      rounded > (double)std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string(key) + " is outside int range");
  }
  return (int)rounded;
}

inline int mtl_validate_nonzero_config_value(const char *key, int value) {
  if (value == 0) {
    throw std::invalid_argument(std::string(key) + " must be nonzero");
  }
  return value;
}

inline int mtl_validate_positive_config_value(const char *key, int value) {
  if (value <= 0) {
    throw std::invalid_argument(std::string(key) + " must be positive");
  }
  return value;
}

inline void mtl_validate_divisible_config_values(const char *numerator_key,
                                                 int numerator,
                                                 const char *denominator_key,
                                                 int denominator) {
  mtl_validate_nonzero_config_value(denominator_key, denominator);
  if (numerator % denominator != 0) {
    throw std::invalid_argument(std::string(numerator_key) + " must be divisible by " +
                                denominator_key + ": " +
                                std::to_string(numerator) + " % " +
                                std::to_string(denominator) + " != 0");
  }
}

inline void mtl_set_tensor(MetalStream *ms, const FloatTensor &t,
                           uint32_t index) {
  NSUInteger offset;
  id<MTLBuffer> buf = mtl_buffer_for_ptr(t.data, &offset);
  uint64_t addr = buf.gpuAddress + offset;
  if (ms->bound_addresses[index] != addr) {
    [ms->arg_table setAddress:addr atIndex:index];
    ms->bound_addresses[index] = addr;
  }
}
template <typename T>
inline void mtl_set_params(MetalStream *ms, const T &params, uint32_t index) {
  NSUInteger next_offset = 0;
  if (!mtl_const_ring_reserve_range(ms->const_ring_offset, sizeof(T),
                                    &next_offset)) {
    std::fprintf(stderr,
                 "mtl_set_params: constant ring overflow: offset=%llu size=%llu capacity=%llu\n",
                 (unsigned long long)ms->const_ring_offset,
                 (unsigned long long)sizeof(T),
                 (unsigned long long)MTL_CONST_RING_SIZE);
    std::abort();
  }
  memcpy((char *)[ms->const_ring contents] + ms->const_ring_offset,
         &params, sizeof(T));
  uint64_t addr = ms->const_ring.gpuAddress + ms->const_ring_offset;
  [ms->arg_table setAddress:addr atIndex:index];
  ms->bound_addresses[index] = addr;
  ms->const_ring_offset = next_offset;
}

inline void mtl_dispatch_1d(MetalStream *ms, id<MTLComputePipelineState> pso,
                            int count) {
  NSUInteger tg = MIN((NSUInteger)pso.maxTotalThreadsPerThreadgroup, 256);
  [ms->enc dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

inline void mtl_dispatch_groups(MetalStream *ms,
                                id<MTLComputePipelineState> pso,
                                int num_groups, int group_size) {
  [ms->enc dispatchThreadgroups:MTLSizeMake(num_groups, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(group_size, 1, 1)];
}

inline void mtl_set_threadgroup_memory(MetalStream *ms, NSUInteger length,
                                       uint32_t index) {
  [ms->enc setThreadgroupMemoryLength:length atIndex:index];
}

inline void mtl_barrier(MetalStream *ms) {
  if (ms->enc_active) {
    [ms->enc barrierAfterEncoderStages:MTLStageDispatch
                    beforeEncoderStages:MTLStageDispatch
                      visibilityOptions:MTL4VisibilityOptionDevice];
    ms->pending_work = true;
  }
}

inline PufTensor to_puf(PrecisionTensor &t) {
  return {.bytes = (char *)t.data,
          .shape = {t.shape[0], t.shape[1], t.shape[2], t.shape[3]},
          .dtype_size = t.dtype_size};
}

inline PufTensor to_puf(const PrecisionTensor &t) {
  return {.bytes = (char *)t.data,
          .shape = {t.shape[0], t.shape[1], t.shape[2], t.shape[3]},
          .dtype_size = t.dtype_size};
}

void puf_mm(PufTensor &a, PufTensor &b, PufTensor &out, cudaStream_t stream);
void puf_mm_tn(PufTensor &a, PufTensor &b, PufTensor &out,
               cudaStream_t stream);
void puf_mm_nn(PufTensor &a, PufTensor &b, PufTensor &out,
               cudaStream_t stream);
void puf_addmm_nn(PufTensor &a, PufTensor &b, PufTensor &out, float alpha,
                   float beta, cudaStream_t stream);
void mtl_sync_stats(int *out_count, double *out_total_ms);
void mtl_enable_gpu_timing(bool enable);
void mtl_gpu_timing_stats(double *gpu_exec_ms, double *sched_wait_ms);
void mtl_gemm_stats(int *tensor_ops_count);
bool puf_stream_has_encoder(cudaStream_t stream);
void puf_set_gpu_training(bool val);
bool puf_is_gpu_training();
void cpu_cast_u8_to_f32(float *dst, const uint8_t *src, int count);
void mtl_cast_f32_to_f16(void *dst, const float *src, int count,
                          cudaStream_t stream);
void mtl_cast_f16_to_f32(float *dst, const void *src, int count,
                          cudaStream_t stream);

#endif // PUFFERLIB_METAL_PLATFORM_H
