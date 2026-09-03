#ifndef OSRS_ASSETS_H
#define OSRS_ASSETS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline double osrs_now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1000.0 + (double)t.tv_nsec / 1e6;
}

static inline void osrs_time_log(const char* label, double* t0) {
    double now = osrs_now_ms();
    if (t0 && *t0 > 0.0) {
        fprintf(stderr, "TIME %-36s %7.1f ms\n", label, now - *t0);
    }
    if (t0) *t0 = now;
}

#include "osrs_assets_generated.h"
#include "osrs_binary_io.h"

#define OSRS_ASSET_ROOT_DEFAULT "ocean/osrs/data"
#define OSRS_ASSET(path) osrs_asset_path(path)

typedef struct {
    unsigned char* data;
    size_t size;
} OsrsAssetBytes;

static inline const char* osrs_asset_root(void) {
    const char* root = getenv("OSRS_ASSET_ROOT");
    return root && root[0] ? root : OSRS_ASSET_ROOT_DEFAULT;
}

static inline int osrs_asset_has_prefix(const char* s, const char* prefix) {
    size_t n = strlen(prefix);
    return strncmp(s, prefix, n) == 0;
}

static inline int osrs_asset_is_absolute_path(const char* path) {
    return path && path[0] == '/';
}

static inline int osrs_asset_manifest_path_is_valid(const char* path) {
    if (!path || !path[0]) return 0;
    if (osrs_asset_is_absolute_path(path)) return 0;
    if (strchr(path, '\\')) return 0;

    const char* part = path;
    while (*part) {
        const char* slash = strchr(part, '/');
        size_t len = slash ? (size_t)(slash - part) : strlen(part);
        if (len == 0) return 0;
        if (len == 1 && part[0] == '.') return 0;
        if (len == 2 && part[0] == '.' && part[1] == '.') return 0;
        if (!slash) return 1;
        part = slash + 1;
    }
    return 0;
}

static inline const char* osrs_asset_logical_path(const char* path) {
    if (!path) return "";
    while (osrs_asset_has_prefix(path, "./")) path += 2;
    const char* root = osrs_asset_root();
    size_t root_len = strlen(root);
    if (root_len > 0 && strncmp(path, root, root_len) == 0 &&
            (path[root_len] == '/' || path[root_len] == '\0')) {
        path += root_len;
        if (path[0] == '/') path++;
    }
    if (osrs_asset_has_prefix(path, "data/")) path += 5;
    const char* data_part = strstr(path, "/data/");
    if (data_part) path = data_part + 6;
    return path;
}

static inline const char* osrs_asset_path(const char* path) {
    enum { OSRS_ASSET_PATH_RING = 64, OSRS_ASSET_PATH_MAX = 2048 };
    static char paths[OSRS_ASSET_PATH_RING][OSRS_ASSET_PATH_MAX];
    static int path_idx = 0;

    if (!path) {
        fprintf(stderr, "osrs_asset_path: path is null\n");
        abort();
    }

    int idx = path_idx++ % OSRS_ASSET_PATH_RING;
    if (osrs_asset_is_absolute_path(path)) {
        int n = snprintf(paths[idx], sizeof(paths[idx]), "%s", path);
        if (n < 0 || (size_t)n >= sizeof(paths[idx])) {
            fprintf(stderr, "osrs_asset_path: absolute path too long: %s\n", path);
            abort();
        }
        return paths[idx];
    }

    const char* logical = osrs_asset_logical_path(path);
    int n = snprintf(paths[idx], sizeof(paths[idx]), "%s/%s",
        osrs_asset_root(), logical);
    if (n < 0 || (size_t)n >= sizeof(paths[idx])) {
        fprintf(stderr, "osrs_asset_path: path too long under %s: %s\n",
            osrs_asset_root(), logical);
        abort();
    }
    return paths[idx];
}

static inline FILE* osrs_asset_fopen(const char* path, const char* mode) {
    return fopen(osrs_asset_path(path), mode);
}

static inline int osrs_asset_exists(const char* path) {
    FILE* f = osrs_asset_fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static inline void osrs_asset_require_group(OsrsAssetGroupKind kind) {
    if ((size_t)kind >= OSRS_ASSET_GROUP_COUNT) {
        fprintf(stderr, "osrs_asset_require_group: bad group: %d\n", (int)kind);
        abort();
    }
    const OsrsAssetGroup* group = &OSRS_ASSET_GROUPS[kind];

    size_t invalid_count = 0;
    size_t missing_count = 0;
    for (size_t i = 0; i < group->path_count; i++) {
        const char* path = group->paths[i];
        if (!osrs_asset_manifest_path_is_valid(path)) {
            if (invalid_count == 0) {
                fprintf(stderr, "OSRS asset group '%s' has invalid manifest paths:\n",
                    group->name);
            }
            fprintf(stderr, "  %s\n", path ? path : "(null)");
            invalid_count++;
            continue;
        }
        if (!osrs_asset_exists(path)) {
            if (missing_count == 0) {
                fprintf(stderr, "OSRS asset group '%s' is missing files under %s:\n",
                    group->name, osrs_asset_root());
            }
            fprintf(stderr, "  %s\n", osrs_asset_path(path));
            missing_count++;
        }
    }

    if (invalid_count || missing_count) {
        fprintf(stderr,
            "OSRS asset group '%s' failed validation: %zu invalid, %zu missing\n",
            group->name, invalid_count, missing_count);
        abort();
    }
}

static inline OsrsAssetBytes osrs_asset_read_all(const char* path) {
    OsrsAssetBytes out = {0};
    const char* full_path = osrs_asset_path(path);
    FILE* f = osrs_asset_fopen(path, "rb");
    if (!f) return out;
    long size = osrs_file_size_or_abort(f, full_path);
    if (size == 0) {
        fclose(f);
        return out;
    }

    out.data = (unsigned char*)malloc((size_t)size);
    if (!out.data) {
        fprintf(stderr, "%s: malloc failed for %ld bytes\n", full_path, size);
        abort();
    }
    out.size = (size_t)size;
    size_t got = fread(out.data, 1, out.size, f);
    if (got != out.size) {
        fprintf(stderr, "%s: short read (%zu/%zu)\n", full_path, got, out.size);
        abort();
    }
    fclose(f);
    return out;
}

static inline void osrs_asset_bytes_free(OsrsAssetBytes* bytes) {
    if (!bytes) return;
    free(bytes->data);
    bytes->data = NULL;
    bytes->size = 0;
}

#endif
