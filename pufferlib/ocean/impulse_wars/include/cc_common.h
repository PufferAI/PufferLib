/*
 * Collections-C
 * Copyright (C) 2013-2014 Srđan Panić <srdja.panic@gmail.com>
 *
 * This file is part of Collections-C.
 *
 * Collections-C is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Collections-C is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with Collections-C.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef CC_COMMON_H
#define CC_COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef ARCH_64
#define MAX_POW_TWO (((size_t)1) << 63)
#else
#define MAX_POW_TWO (((size_t)1) << 31)
#endif /* ARCH_64 */

enum cc_stat {
    CC_OK = 0,

    CC_ERR_ALLOC = 1,
    CC_ERR_INVALID_CAPACITY = 2,
    CC_ERR_INVALID_RANGE = 3,
    CC_ERR_MAX_CAPACITY = 4,
    CC_ERR_KEY_NOT_FOUND = 6,
    CC_ERR_VALUE_NOT_FOUND = 7,
    CC_ERR_OUT_OF_RANGE = 8,

    CC_ITER_END = 9,
};

#define CC_MAX_ELEMENTS ((size_t) - 2)

#if defined(_MSC_VER)

#define INLINE __inline
#define FORCE_INLINE __forceinline

#else

#define INLINE inline
#define FORCE_INLINE inline __attribute__((always_inline))

#endif /* _MSC_VER */

int cc_common_cmp_str(const void *key1, const void *key2);

#define CC_CMP_STRING cc_common_cmp_str

#ifdef __cplusplus
}
#endif

#endif
