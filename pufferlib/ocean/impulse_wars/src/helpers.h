#ifndef IMPULSE_WARS_HELPERS_H
#define IMPULSE_WARS_HELPERS_H

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "box2d/box2d.h"

#include "include/cc_array.h"

#ifndef NDEBUG
#define ON_ERROR __builtin_trap()
#define _DEBUG_GET_TIMEINFO() \
    time_t _t = time(NULL);   \
    struct tm *_timeinfo;     \
    _timeinfo = localtime(&_t)
#define DEBUG_RAW_LOG(msg) \
    do {                   \
        printf(msg);       \
        fflush(stdout);    \
    } while (0)
#define DEBUG_RAW_LOGF(fmt, args...) \
    do {                             \
        printf(fmt, args);           \
        fflush(stdout);              \
    } while (0)
#define DEBUG_LOGF(fmt, args...)                                                                                             \
    do {                                                                                                                     \
        _DEBUG_GET_TIMEINFO();                                                                                               \
        printf(fmt " %d:%d:%d %s:%d\n", args, _timeinfo->tm_hour, _timeinfo->tm_min, _timeinfo->tm_sec, __FILE__, __LINE__); \
        fflush(stdout);                                                                                                      \
    } while (0)
#define DEBUG_LOG(msg)                                                                                                 \
    do {                                                                                                               \
        _DEBUG_GET_TIMEINFO();                                                                                         \
        printf(msg " %d:%d:%d %s:%d\n", _timeinfo->tm_hour, _timeinfo->tm_min, _timeinfo->tm_sec, __FILE__, __LINE__); \
        fflush(stdout);                                                                                                \
    } while (0)

#define ASSERT(condition)                                                                  \
    do {                                                                                   \
        if (!(condition)) {                                                                \
            printf("\nASSERTION FAILED: %s at %s:%d\n\n", #condition, __FILE__, __LINE__); \
            fflush(stdout);                                                                \
            ON_ERROR;                                                                      \
        }                                                                                  \
    } while (0)
#define ASSERTF(condition, fmt, args...)                                                                   \
    do {                                                                                                   \
        if (!(condition)) {                                                                                \
            printf("\nASSERTION FAILED: %s; " fmt "; at %s:%d\n\n", #condition, args, __FILE__, __LINE__); \
            fflush(stdout);                                                                                \
            ON_ERROR;                                                                                      \
        }                                                                                                  \
    } while (0)
#else
#define ON_ERROR abort()
#define DEBUG_RAW_LOG(msg)
#define DEBUG_RAW_LOGF(fmt, args...)
#define DEBUG_LOGF(fmt, args...)
#define DEBUG_LOG(msg)
#define ASSERT(condition)
#define ASSERTF(condition, fmt, args...)
#endif

#define ERRORF(fmt, args...)                                             \
    fprintf(stderr, "FATAL: " fmt " %s:%d\n", args, __FILE__, __LINE__); \
    fflush(stderr);                                                      \
    ON_ERROR
#define ERROR(msg)                                                 \
    fprintf(stderr, "FATAL: " msg " %s:%d\n", __FILE__, __LINE__); \
    fflush(stderr);                                                \
    ON_ERROR

// ignore compiler warnings about unused variables for variables that are
// only used in debug builds
#define MAYBE_UNUSED(x) (void)x

#define SQUARED(x) ((x) * (x))

#ifndef PI
#define PI 3.14159265358979323846f
#endif

#ifndef RAD2DEG
#define RAD2DEG (180.0f / PI)
#endif

#define MASS(density, radius) ((density) * PI * (radius) * (radius))
#define INV_MASS(mass) (1.0f / (mass))
// given a mass and radius, calculate the density such that the density
// and radius will result in the same mass as what's given
#define MATCHING_DENSITY(mass, radius) ((mass) / (PI * (radius) * (radius)))

#define ASSERT_VEC_BOUNDED(vec)                  \
    ASSERTF(vec.x <= 1.0f, "vec.x: %f", vec.x);  \
    ASSERTF(vec.x >= -1.0f, "vec.x: %f", vec.x); \
    ASSERTF(vec.y <= 1.0f, "vec.y: %f", vec.y);  \
    ASSERTF(vec.y >= -1.0f, "vec.y: %f", vec.y)

#define ASSERT_VEC_NORMALIZED(vec)                                                                           \
    ASSERT_VEC_BOUNDED(vec);                                                                                 \
    do {                                                                                                     \
        const b2Vec2 norm = b2Normalize(vec);                                                                \
        MAYBE_UNUSED(norm);                                                                                  \
        ASSERTF(fabs(vec.x - norm.x) < 0.000001f, "vec: %f, %f norm: %f, %f", vec.x, vec.y, norm.x, norm.y); \
        ASSERTF(fabs(vec.y - norm.y) < 0.000001f, "vec: %f, %f norm: %f, %f", vec.x, vec.y, norm.x, norm.y); \
    } while (0)

// use malloc when debugging so the address sanitizer can find issues with
// heap memory, use dlmalloc in release mode for performance; emscripten
// uses dlmalloc by default so no need to change anything here
#if !defined(NDEBUG) || defined(__EMSCRIPTEN__)
#define fastMalloc(size) malloc(size)
#define fastMallocFn malloc
#define fastCalloc(nmemb, size) calloc(nmemb, size)
#define fastCallocFn calloc
#define fastFree(ptr) free(ptr)
#define fastFreeFn free
#else
#include "include/dlmalloc.h"
#define fastMalloc(size) dlmalloc(size)
#define fastMallocFn dlmalloc
#define fastCalloc(nmemb, size) dlcalloc(nmemb, size)
#define fastCallocFn dlcalloc
#define fastFree(ptr) dlfree(ptr)
#define fastFreeFn dlfree
#endif

static inline void create_array(CC_Array **array, size_t initialCap) {
    CC_ArrayConf conf;
    cc_array_conf_init(&conf);
    conf.capacity = initialCap;
    conf.mem_alloc = fastMallocFn;
    conf.mem_calloc = fastCallocFn;
    conf.mem_free = fastFreeFn;

    cc_array_new_conf(&conf, array);
}

// automatically checks that the index is valid and returns the value
// so callers can use it as a constant expression
static inline void *safe_array_get_at(const CC_Array *const array, size_t index) {
    void *val;
    const enum cc_stat res = cc_array_get_at(array, index, &val);
    ASSERT(res == CC_OK);
    MAYBE_UNUSED(res);
    return val;
}

static inline bool b2VecEqual(const b2Vec2 v1, const b2Vec2 v2) {
    return v1.x == v2.x && v1.y == v2.y;
}

#ifndef AUTOPXD
// from https://lemire.me/blog/2019/03/19/the-fastest-conventional-random-number-generator-that-can-pass-big-crush/
// see also https://github.com/lemire/testingRNG
uint64_t wyhash64(uint64_t *state) {
    *state += 0x60bee2bee120fc15;
    __uint128_t tmp;
    tmp = (__uint128_t)(*state) * 0xa3b195354a39b70d;
    uint64_t m1 = (tmp >> 64) ^ tmp;
    tmp = (__uint128_t)m1 * 0x1b03738712fad5c9;
    uint64_t m2 = (tmp >> 64) ^ tmp;
    return m2;
}
#endif

static inline float randFloat(uint64_t *state, const float min, const float max) {
    float n = wyhash64(state) / (float)UINT64_MAX;
    return min + n * (max - min);
}

static inline int randInt(uint64_t *state, const int min, const int max) {
    return min + wyhash64(state) % (max - min + 1);
}

static inline float logBasef(const float v, const float b) {
    return log2f(v) / log2(b);
}

#ifndef AUTOPXD
#define min(a, b)               \
    ({                          \
        __typeof__(a) _a = (a); \
        __typeof__(b) _b = (b); \
        _a < _b ? _a : _b;      \
    })

#define max(a, b)               \
    ({                          \
        __typeof__(a) _a = (a); \
        __typeof__(b) _b = (b); \
        _a > _b ? _a : _b;      \
    })
#endif

// clamps between 0 and 1
static inline float clamp(float f) {
    return min(max(f, 0.0f), 1.0f);
}

// normalize value to be between 0 and max, or -max and max;
// minIsZero determines if the min value is 0 or -max
static inline float scaleValue(const float v, const float max, const bool minIsZero) {
    ASSERTF(v <= max, "v: %f, max: %f", v, max);
    ASSERTF(!minIsZero || v >= 0, "v: %f", v);
    ASSERTF(minIsZero || v >= -max, "v: %f, -max: %f", v, -max);

    float scaled = v / max;
    if (minIsZero) {
        return max(min(scaled, max), 0.0f);
    } else {
        return max(min(scaled, max), -1.0f);
    }
}

static inline uint8_t oneHotEncode(float *obs, const uint16_t offset, const uint8_t val, const uint8_t max) {
    ASSERTF(val < max, "val: %d, max: %d", val, max);
    memset(obs + offset, 0x0, max * sizeof(float));
    obs[offset + val] = 1;
    return max;
}

static inline uint16_t alignedSize(const uint16_t size, const uint8_t align) {
    return (size + align - 1) & ~(align - 1);
}

#define BITNSLOTS(nb) ((nb + sizeof(uint8_t) - 1) / sizeof(uint8_t))

static inline uint16_t bitMask(const uint16_t n) {
    return 1 << (n % sizeof(uint8_t));
}

static inline uint16_t bitSlot(const uint16_t n) {
    return n / sizeof(uint8_t);
}

static inline void bitSet(uint8_t *b, const uint16_t n) {
    b[bitSlot(n)] |= bitMask(n);
}

static inline bool bitTest(const uint8_t *b, const uint16_t n) {
    return b[bitSlot(n)] & bitMask(n);
}

#endif
