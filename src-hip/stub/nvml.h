// Minimal NVML stub for AMD builds - utilization stats read as zeros.
#ifndef STUB_NVML_H
#define STUB_NVML_H
#include <stddef.h>
typedef struct nvmlDevice_st* nvmlDevice_t;
typedef struct { unsigned int gpu; unsigned int memory; } nvmlUtilization_t;
typedef struct { unsigned long long total; unsigned long long free; unsigned long long used; } nvmlMemory_t;
#ifndef STUB_NVML_FREE_MEMBER
#define STUB_NVML_FREE_MEMBER
#endif
static inline int nvmlInit_v2(void) { return 0; }
static inline int nvmlInit(void) { return 0; }
static inline int nvmlShutdown(void) { return 0; }
static inline int nvmlDeviceGetHandleByIndex(unsigned int i, nvmlDevice_t* d) { (void)i; *d = 0; return 0; }
static inline int nvmlDeviceGetUtilizationRates(nvmlDevice_t d, nvmlUtilization_t* u) { (void)d; u->gpu = 0; u->memory = 0; return 0; }
static inline int nvmlDeviceGetMemoryInfo(nvmlDevice_t d, nvmlMemory_t* m) { (void)d; m->total = 0; m->used = 0; m->free = 0; return 0; }
#endif
// NVTX no-ops for AMD builds
static inline int nvtxRangePushA(const char* name) { (void)name; return 0; }
static inline void nvtxRangePop(void) {}
