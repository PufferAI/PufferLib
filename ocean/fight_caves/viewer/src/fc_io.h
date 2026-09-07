#ifndef FC_IO_H
#define FC_IO_H

#include <stdbool.h>
#include <stdio.h>

static inline bool fc_read_exact(FILE* f, void* dst, size_t elem_size,
                                 size_t elem_count, const char* path,
                                 const char* what) {
    if (fread(dst, elem_size, elem_count, f) != elem_count) {
        fprintf(stderr, "%s: short read while loading %s\n", path, what);
        return false;
    }
    return true;
}

static inline bool fc_seek(FILE* f, long offset, int origin,
                           const char* path, const char* what) {
    if (fseek(f, offset, origin) != 0) {
        fprintf(stderr, "%s: seek failed while loading %s\n", path, what);
        return false;
    }
    return true;
}

#endif /* FC_IO_H */
