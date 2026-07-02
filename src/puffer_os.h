// puffer_os.h - portability shim for POSIX APIs used by PufferLib.
//
// On POSIX platforms this is a pass-through to the real headers. On Windows it
// emulates the small POSIX surface the codebase uses (pthread create/join,
// clock_gettime, sleep/usleep, access) so call sites stay unchanged.
//
// Deliberately does NOT include windows.h: raylib and windows.h have symbol
// clashes (Rectangle, CloseWindow, ShowCursor, ...). The few Win32 imports
// needed are declared by hand instead.

#ifndef PUFFER_OS_H
#define PUFFER_OS_H

#include <time.h>
#include <stdio.h>

#ifndef _WIN32
// ============================================================================
// POSIX: real headers
// ============================================================================
#include <unistd.h>
#include <pthread.h>

#else
// ============================================================================
// Windows
// ============================================================================
#include <io.h>       // _access
#include <malloc.h>   // _aligned_malloc
#include <process.h>  // _beginthreadex
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// glibc leaks this legacy typedef via sys/types.h; some sources rely on it.
// Linux 'unsigned long' is 64-bit, so match that width here.
typedef unsigned long long ulong;

// Hand-declared Win32 imports (see header comment for why not windows.h).
// When windows.h is already included (e.g. via CUDA's nvml.h), use its
// declarations instead: redeclaring with different pointer types is an error.
#ifdef _WINDOWS_
#define PUFFER_QPC(p)  QueryPerformanceCounter((LARGE_INTEGER*)(p))
#define PUFFER_QPF(p)  QueryPerformanceFrequency((LARGE_INTEGER*)(p))
#define PUFFER_FTIME(p) GetSystemTimePreciseAsFileTime((FILETIME*)(p))
#else
__declspec(dllimport) unsigned long __stdcall WaitForSingleObject(void* handle, unsigned long ms);
__declspec(dllimport) int __stdcall CloseHandle(void* handle);
__declspec(dllimport) void __stdcall Sleep(unsigned long ms);
__declspec(dllimport) int __stdcall QueryPerformanceCounter(long long* count);
__declspec(dllimport) int __stdcall QueryPerformanceFrequency(long long* freq);
__declspec(dllimport) void __stdcall GetSystemTimePreciseAsFileTime(unsigned long long* filetime);
#define PUFFER_QPC(p)  QueryPerformanceCounter(p)
#define PUFFER_QPF(p)  QueryPerformanceFrequency(p)
#define PUFFER_FTIME(p) GetSystemTimePreciseAsFileTime(p)
#endif

// UCRT stdlib.h defines min/max macros in C mode; glibc does not, and env
// code defines its own inline min/max. Neutralize the macros.
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

// ---- access() ----
#ifndef F_OK
#define F_OK 0
#endif
#ifndef R_OK
#define R_OK 4
#endif
#define access _access

// ---- sleep()/usleep() ----
static __inline void puffer_sleep_ms_(unsigned long ms) { Sleep(ms); }
#define sleep(s) puffer_sleep_ms_(1000ul * (unsigned long)(s))
#define usleep(us) puffer_sleep_ms_((unsigned long)((us) / 1000))

// ---- random() ---- (31-bit; backed by the rand_r shim below)
static __inline int rand_r(unsigned int* seed);
static __inline long random(void) {
    static unsigned int puffer_random_state_ = 0x853c49e6u;
    return (long)rand_r(&puffer_random_state_);
}

// ---- rand_r() ---- (glibc's algorithm, for distribution parity)
static __inline int rand_r(unsigned int* seed) {
    unsigned int next = *seed;
    int result;
    next = next * 1103515245u + 12345u;
    result = (int)((next / 65536u) % 2048u);
    next = next * 1103515245u + 12345u;
    result <<= 10;
    result ^= (int)((next / 65536u) % 1024u);
    next = next * 1103515245u + 12345u;
    result <<= 10;
    result ^= (int)((next / 65536u) % 1024u);
    *seed = next;
    return result;
}

// ---- clock_gettime() ----
#ifndef CLOCK_REALTIME
#define CLOCK_REALTIME 0
#endif
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif

static __inline int clock_gettime(int clock_id, struct timespec* ts) {
    if (clock_id == CLOCK_MONOTONIC) {
        static long long freq = 0;
        long long count;
        if (freq == 0) PUFFER_QPF(&freq);
        PUFFER_QPC(&count);
        ts->tv_sec = (time_t)(count / freq);
        ts->tv_nsec = (long)((count % freq) * 1000000000ll / freq);
        return 0;
    }
    // CLOCK_REALTIME: FILETIME is 100ns ticks since 1601-01-01
    unsigned long long ft;
    PUFFER_FTIME(&ft);
    ft -= 116444736000000000ull;  // to Unix epoch
    ts->tv_sec = (time_t)(ft / 10000000ull);
    ts->tv_nsec = (long)(ft % 10000000ull) * 100;
    return 0;
}

// ---- minimal pthreads (create/join only, as used by src/vecenv.h) ----
typedef uintptr_t pthread_t;

typedef struct puffer_thread_tramp_ {
    void* (*fn)(void*);
    void* arg;
} puffer_thread_tramp_;

static unsigned __stdcall puffer_thread_entry_(void* p) {
    puffer_thread_tramp_ t = *(puffer_thread_tramp_*)p;
    free(p);
    t.fn(t.arg);
    return 0;
}

static __inline int pthread_create(pthread_t* thread, const void* attr,
        void* (*fn)(void*), void* arg) {
    (void)attr;
    puffer_thread_tramp_* t = (puffer_thread_tramp_*)malloc(sizeof(*t));
    if (!t) return -1;
    t->fn = fn;
    t->arg = arg;
    *thread = _beginthreadex(NULL, 0, puffer_thread_entry_, t, 0, NULL);
    if (*thread == 0) {
        free(t);
        return -1;
    }
    return 0;
}

static __inline int pthread_join(pthread_t thread, void** retval) {
    (void)retval;
    WaitForSingleObject((void*)thread, 0xFFFFFFFFul);  // INFINITE
    CloseHandle((void*)thread);
    return 0;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // _WIN32

// ============================================================================
// Aligned allocation. C11 aligned_alloc memory is released with plain free(),
// which MSVC's UCRT cannot do (_aligned_malloc requires _aligned_free), so
// call sites needing aligned memory use this pair instead.
// ============================================================================
#ifdef _WIN32
#define puffer_aligned_alloc(alignment, size) _aligned_malloc((size), (alignment))
#define puffer_aligned_free _aligned_free
#else
#define puffer_aligned_alloc(alignment, size) aligned_alloc((alignment), (size))
#define puffer_aligned_free free
#endif

// ============================================================================
// Resident memory (RSS) in kilobytes; 0 if unavailable. All platforms.
// ============================================================================
#ifdef _WIN32
#ifdef __cplusplus
extern "C" {
#endif
typedef struct puffer_pmc_ {
    unsigned long cb;
    unsigned long PageFaultCount;
    size_t PeakWorkingSetSize;
    size_t WorkingSetSize;
    size_t QuotaPeakPagedPoolUsage;
    size_t QuotaPagedPoolUsage;
    size_t QuotaPeakNonPagedPoolUsage;
    size_t QuotaNonPagedPoolUsage;
    size_t PagefileUsage;
    size_t PeakPagefileUsage;
} puffer_pmc_;
__declspec(dllimport) void* __stdcall GetCurrentProcess(void);
__declspec(dllimport) int __stdcall K32GetProcessMemoryInfo(void* process, puffer_pmc_* counters, unsigned long cb);

static __inline long puffer_resident_kb(void) {
    puffer_pmc_ pmc;
    pmc.cb = (unsigned long)sizeof(pmc);
    if (!K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, pmc.cb)) return 0;
    return (long)(pmc.WorkingSetSize / 1024);
}
#ifdef __cplusplus
}  // extern "C"
#endif
#else
static inline long puffer_resident_kb(void) {
    long rss_kb = 0;
    FILE* f = fopen("/proc/self/status", "r");  // Linux; absent elsewhere -> 0
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (sscanf(line, "VmRSS: %ld", &rss_kb) == 1) break;
        }
        fclose(f);
    }
    return rss_kb;
}
#endif

#endif  // PUFFER_OS_H
