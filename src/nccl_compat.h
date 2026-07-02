// nccl_compat.h - NCCL include indirection.
//
// NCCL has no Windows build, so multi-GPU support is compile-time optional:
// CMake defines PUFFER_HAS_NCCL when NCCL is found (Linux). Without it, this
// header provides just enough for single-GPU code paths to compile; all
// actual NCCL calls are guarded by #ifdef PUFFER_HAS_NCCL at their call sites,
// and requesting world_size > 1 raises a runtime error instead.

#ifndef PUFFER_NCCL_COMPAT_H
#define PUFFER_NCCL_COMPAT_H

#ifdef PUFFER_HAS_NCCL
#include <nccl.h>
#else
typedef void* ncclComm_t;
#endif

#endif  // PUFFER_NCCL_COMPAT_H
