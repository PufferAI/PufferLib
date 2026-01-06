/*
 * Collections-C
 * Copyright (C) 2013-2015 Srđan Panić <srdja.panic@gmail.com>
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

#ifndef CC_ARRAY_H
#define CC_ARRAY_H

#ifdef __cplusplus
extern "C" {
#endif

#include "cc_common.h"

/**
 * A dynamic array that expands automatically as elements are
 * added. The array supports amortized constant time insertion
 * and removal of elements at the end of the array, as well as
 * constant time access.
 */
typedef struct cc_array_s CC_Array;

/**
 * Array configuration structure. Used to initialize a new Array
 * with specific values.
 */
typedef struct cc_array_conf_s {
    /**
     * The initial capacity of the array */
    size_t capacity;

    /**
     * The rate at which the buffer expands (capacity * exp_factor). */
    float exp_factor;

    /**
     * Memory allocators used to allocate the Array structure and the
     * underlying data buffers. */
    void *(*mem_alloc)(size_t size);
    void *(*mem_calloc)(size_t blocks, size_t size);
    void (*mem_free)(void *block);
} CC_ArrayConf;

/**
 * Array iterator structure. Used to iterate over the elements of
 * the array in an ascending order. The iterator also supports
 * operations for safely adding and removing elements during
 * iteration.
 */
typedef struct cc_array_iter_s {
    /**
     * The array associated with this iterator */
    CC_Array *ar;

    /**
     * The current position of the iterator.*/
    size_t index;

    /**
     * Set to true if the last returned element was removed. */
    bool last_removed;
} CC_ArrayIter;

/**
 * Array zip iterator structure. Used to iterate over the elements of two
 * arrays in lockstep in an ascending order until one of the Arrays is
 * exhausted. The iterator also supports operations for safely adding
 * and removing elements during iteration.
 */
typedef struct array_zip_iter_s {
    CC_Array *ar1;
    CC_Array *ar2;
    size_t index;
    bool last_removed;
} CC_ArrayZipIter;

enum cc_stat cc_array_new(CC_Array **out);
enum cc_stat cc_array_new_conf(CC_ArrayConf const *const conf, CC_Array **out);
void cc_array_conf_init(CC_ArrayConf *conf);
size_t cc_array_struct_size();

void cc_array_destroy(CC_Array *ar);
void cc_array_destroy_cb(CC_Array *ar, void (*cb)(void *));

enum cc_stat cc_array_add(CC_Array *ar, void *element);
enum cc_stat cc_array_add_at(CC_Array *ar, void *element, size_t index);
enum cc_stat cc_array_replace_at(CC_Array *ar, void *element, size_t index, void **out);
enum cc_stat cc_array_swap_at(CC_Array *ar, size_t index1, size_t index2);

enum cc_stat cc_array_remove(CC_Array *ar, void *element, void **out);
enum cc_stat cc_array_remove_fast(CC_Array *ar, void *element, void **out);
enum cc_stat cc_array_remove_at(CC_Array *ar, size_t index, void **out);
enum cc_stat cc_array_remove_fast_at(CC_Array *ar, size_t index, void **out);
enum cc_stat cc_array_remove_last(CC_Array *ar, void **out);
void cc_array_remove_all(CC_Array *ar);
void cc_array_remove_all_free(CC_Array *ar);

enum cc_stat cc_array_get_at(const CC_Array *ar, size_t index, void **out);
enum cc_stat cc_array_get_last(const CC_Array *ar, void **out);

enum cc_stat cc_array_subarray(CC_Array *ar, size_t from, size_t to, CC_Array **out);
enum cc_stat cc_array_copy_shallow(CC_Array *ar, CC_Array **out);
enum cc_stat cc_array_copy_deep(CC_Array *ar, void *(*cp)(void *), CC_Array **out);

void cc_array_reverse(CC_Array *ar);
enum cc_stat cc_array_trim_capacity(CC_Array *ar);

size_t cc_array_contains(const CC_Array *ar, void *element);
size_t cc_array_contains_value(const CC_Array *ar, void *element, int (*cmp)(const void *, const void *));
size_t cc_array_size(const CC_Array *ar);
size_t cc_array_capacity(const CC_Array *ar);

enum cc_stat cc_array_index_of(const CC_Array *ar, void *element, size_t *index);
void cc_array_sort(CC_Array *ar, int (*cmp)(const void *, const void *));

void cc_array_map(CC_Array *ar, void (*fn)(void *));
void cc_array_reduce(CC_Array *ar, void (*fn)(void *, void *, void *), void *result);

enum cc_stat cc_array_filter_mut(CC_Array *ar, bool (*predicate)(const void *));
enum cc_stat cc_array_filter(CC_Array *ar, bool (*predicate)(const void *), CC_Array **out);

void cc_array_iter_init(CC_ArrayIter *iter, CC_Array *ar);
enum cc_stat cc_array_iter_next(CC_ArrayIter *iter, void **out);
enum cc_stat cc_array_iter_remove(CC_ArrayIter *iter, void **out);
enum cc_stat cc_array_iter_remove_fast(CC_ArrayIter *iter, void **out);
enum cc_stat cc_array_iter_add(CC_ArrayIter *iter, void *element);
enum cc_stat cc_array_iter_replace(CC_ArrayIter *iter, void *element, void **out);
size_t cc_array_iter_index(CC_ArrayIter *iter);

void cc_array_zip_iter_init(CC_ArrayZipIter *iter, CC_Array *a1, CC_Array *a2);
enum cc_stat cc_array_zip_iter_next(CC_ArrayZipIter *iter, void **out1, void **out2);
enum cc_stat cc_array_zip_iter_add(CC_ArrayZipIter *iter, void *e1, void *e2);
enum cc_stat cc_array_zip_iter_remove(CC_ArrayZipIter *iter, void **out1, void **out2);
enum cc_stat cc_array_zip_iter_replace(CC_ArrayZipIter *iter, void *e1, void *e2, void **out1, void **out2);
size_t cc_array_zip_iter_index(CC_ArrayZipIter *iter);

const void *const *cc_array_get_buffer(CC_Array *ar);

#define CC_ARRAY_FOREACH(val, array, body)                                               \
    {                                                                                    \
        CC_ArrayIter cc_array_iter_53d46d2a04458e7b;                                     \
        cc_array_iter_init(&cc_array_iter_53d46d2a04458e7b, array);                      \
        void *val;                                                                       \
        while (cc_array_iter_next(&cc_array_iter_53d46d2a04458e7b, &val) != CC_ITER_END) \
            body                                                                         \
    }

#define CC_ARRAY_FOREACH_ZIP(val1, val2, array1, array2, body)                                            \
    {                                                                                                     \
        CC_ArrayZipIter cc_array_zip_iter_ea08d3e52f25883b3;                                              \
        cc_array_zip_iter_init(&cc_array_zip_iter_ea08d3e52f25883b3, array1, array2);                     \
        void *val1;                                                                                       \
        void *val2;                                                                                       \
        while (cc_array_zip_iter_next(&cc_array_zip_iter_ea08d3e52f25883b3, &val1, &val2) != CC_ITER_END) \
            body                                                                                          \
    }

#define DEFAULT_CAPACITY 8
#define DEFAULT_EXPANSION_FACTOR 2

struct cc_array_s {
    size_t size;
    size_t capacity;
    float exp_factor;
    void **buffer;

    void *(*mem_alloc)(size_t size);
    void *(*mem_calloc)(size_t blocks, size_t size);
    void (*mem_free)(void *block);
};

static enum cc_stat expand_array_capacity(CC_Array *ar);

/**
 * Creates a new empty array and returns a status code.
 *
 * @param[out] out pointer to where the newly created CC_Array is to be stored
 *
 * @return CC_OK if the creation was successful, or CC_ERR_ALLOC if the
 * memory allocation for the new CC_Array structure failed.
 */
enum cc_stat cc_array_new(CC_Array **out) {
    CC_ArrayConf c;
    cc_array_conf_init(&c);
    return cc_array_new_conf(&c, out);
}

/**
 * Creates a new empty CC_Array based on the specified CC_ArrayConf struct and
 * returns a status code.
 *
 * The CC_Array is allocated using the allocators specified in the CC_ArrayConf
 * struct. The allocation may fail if underlying allocator fails. It may also
 * fail if the values of exp_factor and capacity in the CC_ArrayConf do not meet
 * the following condition: <code>exp_factor < (CC_MAX_ELEMENTS / capacity)</code>.
 *
 * @param[in] conf array configuration structure
 * @param[out] out pointer to where the newly created CC_Array is to be stored
 *
 * @return CC_OK if the creation was successful, CC_ERR_INVALID_CAPACITY if
 * the above mentioned condition is not met, or CC_ERR_ALLOC if the memory
 * allocation for the new CC_Array structure failed.
 */
enum cc_stat cc_array_new_conf(CC_ArrayConf const *const conf, CC_Array **out) {
    float ex;

    /* The expansion factor must be greater than one for the
     * array to grow */
    if (conf->exp_factor <= 1) {
        ex = DEFAULT_EXPANSION_FACTOR;
    } else {
        ex = conf->exp_factor;
    }

    /* Needed to avoid an integer overflow on the first resize and
     * to easily check for any future overflows. */
    if (!conf->capacity || ex >= CC_MAX_ELEMENTS / conf->capacity) {
        return CC_ERR_INVALID_CAPACITY;
    }

    CC_Array *ar = (CC_Array *)conf->mem_calloc(1, sizeof(CC_Array));

    if (!ar) {
        return CC_ERR_ALLOC;
    }

    void **buff = (void **)conf->mem_alloc(conf->capacity * sizeof(void *));

    if (!buff) {
        conf->mem_free(ar);
        return CC_ERR_ALLOC;
    }

    ar->buffer = buff;
    ar->exp_factor = ex;
    ar->capacity = conf->capacity;
    ar->mem_alloc = conf->mem_alloc;
    ar->mem_calloc = conf->mem_calloc;
    ar->mem_free = conf->mem_free;

    *out = ar;
    return CC_OK;
}

/**
 * Initializes the fields of the CC_ArrayConf struct to default values.
 *
 * @param[in, out] conf CC_ArrayConf structure that is being initialized
 */
void cc_array_conf_init(CC_ArrayConf *conf) {
    conf->exp_factor = DEFAULT_EXPANSION_FACTOR;
    conf->capacity = DEFAULT_CAPACITY;
    conf->mem_alloc = malloc;
    conf->mem_calloc = calloc;
    conf->mem_free = free;
}

/**
 * Destroys the CC_Array structure, but leaves the data it used to hold intact.
 *
 * @param[in] ar the array that is to be destroyed
 */
void cc_array_destroy(CC_Array *ar) {
    ar->mem_free(ar->buffer);
    ar->mem_free(ar);
}

/**
 * Destroys the CC_Array structure along with all the data it holds.
 *
 * @note
 * This function should not be called on a array that has some of its elements
 * allocated on the stack.
 *
 * @param[in] ar the array that is being destroyed
 */
void cc_array_destroy_cb(CC_Array *ar, void (*cb)(void *)) {
    size_t i;
    for (i = 0; i < ar->size; i++) {
        cb(ar->buffer[i]);
    }

    cc_array_destroy(ar);
}

/**
 * Adds a new element to the CC_Array. The element is appended to the array making
 * it the last element (the one with the highest index) of the CC_Array.
 *
 * @param[in] ar the array to which the element is being added
 * @param[in] element the element that is being added
 *
 * @return CC_OK if the element was successfully added, CC_ERR_ALLOC if the
 * memory allocation for the new element failed, or CC_ERR_MAX_CAPACITY if the
 * array is already at maximum capacity.
 */
enum cc_stat cc_array_add(CC_Array *ar, void *element) {
    if (ar->size >= ar->capacity) {
        enum cc_stat status = expand_array_capacity(ar);
        if (status != CC_OK) {
            return status;
        }
    }

    ar->buffer[ar->size] = element;
    ar->size++;

    return CC_OK;
}

/**
 * Adds a new element to the array at a specified position by shifting all
 * subsequent elements by one. The specified index must be within the bounds
 * of the array. This function may also fail if the memory allocation for
 * the new element was unsuccessful.
 *
 * @param[in] ar the array to which the element is being added
 * @param[in] element the element that is being added
 * @param[in] index the position in the array at which the element is being
 *            added
 *
 * @return CC_OK if the element was successfully added, CC_ERR_OUT_OF_RANGE if
 * the specified index was not in range, CC_ERR_ALLOC if the memory
 * allocation for the new element failed, or CC_ERR_MAX_CAPACITY if the
 * array is already at maximum capacity.
 */
enum cc_stat cc_array_add_at(CC_Array *ar, void *element, size_t index) {
    if (index == ar->size) {
        return cc_array_add(ar, element);
    }

    if ((ar->size == 0 && index != 0) || index > (ar->size - 1)) {
        return CC_ERR_OUT_OF_RANGE;
    }

    if (ar->size >= ar->capacity) {
        enum cc_stat status = expand_array_capacity(ar);
        if (status != CC_OK) {
            return status;
        }
    }

    size_t shift = (ar->size - index) * sizeof(void *);

    memmove(&(ar->buffer[index + 1]),
            &(ar->buffer[index]),
            shift);

    ar->buffer[index] = element;
    ar->size++;

    return CC_OK;
}

/**
 * Replaces an array element at the specified index and optionally sets the out
 * parameter to the value of the replaced element. The specified index must be
 * within the bounds of the CC_Array.
 *
 * @param[in]  ar      array whose element is being replaced
 * @param[in]  element replacement element
 * @param[in]  index   index at which the replacement element should be inserted
 * @param[out] out     pointer to where the replaced element is stored, or NULL if
 *                     it is to be ignored
 *
 * @return CC_OK if the element was successfully replaced, or CC_ERR_OUT_OF_RANGE
 *         if the index was out of range.
 */
enum cc_stat cc_array_replace_at(CC_Array *ar, void *element, size_t index, void **out) {
    if (index >= ar->size) {
        return CC_ERR_OUT_OF_RANGE;
    }

    if (out) {
        *out = ar->buffer[index];
    }

    ar->buffer[index] = element;

    return CC_OK;
}

enum cc_stat cc_array_swap_at(CC_Array *ar, size_t index1, size_t index2) {
    void *tmp;

    if (index1 >= ar->size || index2 >= ar->size) {
        return CC_ERR_OUT_OF_RANGE;
    }

    tmp = ar->buffer[index1];

    ar->buffer[index1] = ar->buffer[index2];
    ar->buffer[index2] = tmp;
    return CC_OK;
}

/**
 * Removes the specified element from the CC_Array if such element exists and
 * optionally sets the out parameter to the value of the removed element.
 *
 * @param[in] ar array from which the element is being removed
 * @param[in] element element being removed
 * @param[out] out pointer to where the removed value is stored, or NULL
 *                 if it is to be ignored
 *
 * @return CC_OK if the element was successfully removed, or
 * CC_ERR_VALUE_NOT_FOUND if the element was not found.
 */
enum cc_stat cc_array_remove(CC_Array *ar, void *element, void **out) {
    size_t index;
    enum cc_stat status = cc_array_index_of(ar, element, &index);

    if (status == CC_ERR_OUT_OF_RANGE) {
        return CC_ERR_VALUE_NOT_FOUND;
    }

    if (index != ar->size - 1) {
        size_t block_size = (ar->size - 1 - index) * sizeof(void *);

        memmove(&(ar->buffer[index]),
                &(ar->buffer[index + 1]),
                block_size);
    }
    ar->size--;

    if (out) {
        *out = element;
    }

    return CC_OK;
}

/**
 * Removes a CC_Array element without preserving order and optionally sets the
 * out parameter to the value of the removed element. The last element of the
 * array is moved to the index of the element being removed, and the last
 * element is removed.
 *
 * @param[in] ar the array whose last element is being removed
 * @param[out] out pointer to where the removed value is stored, or NULL if it is
 *                 to be ignored
 *
 * @return CC_OK if the element was successfully removed, or CC_ERR_OUT_OF_RANGE
 * if the CC_Array is already empty.
 */
enum cc_stat cc_array_remove_fast(CC_Array *ar, void *element, void **out) {
    size_t index = 0;
    const enum cc_stat status = cc_array_index_of(ar, element, &index);
    if (status != CC_OK) {
        return status;
    }

    if (out) {
        *out = ar->buffer[index];
    }

    ar->buffer[index] = ar->buffer[ar->size - 1];
    ar->size--;

    return CC_OK;
}

/**
 * Removes an CC_Array element from the specified index and optionally sets the
 * out parameter to the value of the removed element. The index must be within
 * the bounds of the array.
 *
 * @param[in] ar the array from which the element is being removed
 * @param[in] index the index of the element being removed.
 * @param[out] out  pointer to where the removed value is stored,
 *                  or NULL if it is to be ignored
 *
 * @return CC_OK if the element was successfully removed, or CC_ERR_OUT_OF_RANGE
 * if the index was out of range.
 */
enum cc_stat cc_array_remove_at(CC_Array *ar, size_t index, void **out) {
    if (index >= ar->size) {
        return CC_ERR_OUT_OF_RANGE;
    }

    if (out) {
        *out = ar->buffer[index];
    }

    if (index != ar->size - 1) {
        size_t block_size = (ar->size - 1 - index) * sizeof(void *);

        memmove(&(ar->buffer[index]),
                &(ar->buffer[index + 1]),
                block_size);
    }
    ar->size--;

    return CC_OK;
}

/**
 * Removes a CC_Array element from the specified index and optionally sets the
 * out parameter to the value of the removed element without preserving ordering.
 * The last element of the array is moved to the index of the element being removed,
 * and the last element is removed. The index must be within the bounds of the array.
 *
 * @param[in] ar the array from which the element is being removed
 * @param[in] index the index of the element being removed.
 * @param[out] out  pointer to where the removed value is stored,
 *                  or NULL if it is to be ignored
 *
 * @return CC_OK if the element was successfully removed, or CC_ERR_OUT_OF_RANGE
 * if the index was out of range.
 */
enum cc_stat cc_array_remove_fast_at(CC_Array *ar, size_t index, void **out) {
    if (index >= ar->size) {
        return CC_ERR_OUT_OF_RANGE;
    }

    if (out) {
        *out = ar->buffer[index];
    }

    ar->buffer[index] = ar->buffer[ar->size - 1];
    ar->size--;

    return CC_OK;
}

/**
 * Removes an CC_Array element from the end of the array and optionally sets the
 * out parameter to the value of the removed element.
 *
 * @param[in] ar the array whose last element is being removed
 * @param[out] out pointer to where the removed value is stored, or NULL if it is
 *                 to be ignored
 *
 * @return CC_OK if the element was successfully removed, or CC_ERR_OUT_OF_RANGE
 * if the CC_Array is already empty.
 */
enum cc_stat cc_array_remove_last(CC_Array *ar, void **out) {
    return cc_array_remove_at(ar, ar->size - 1, out);
}

/**
 * Removes all elements from the specified array. This function does not shrink
 * the array capacity.
 *
 * @param[in] ar array from which all elements are to be removed
 */
void cc_array_remove_all(CC_Array *ar) {
    ar->size = 0;
}

/**
 * Removes and frees all elements from the specified array. This function does
 * not shrink the array capacity.
 *
 * @param[in] ar array from which all elements are to be removed
 */
void cc_array_remove_all_free(CC_Array *ar) {
    size_t i;
    for (i = 0; i < ar->size; i++) {
        free(ar->buffer[i]);
    }

    cc_array_remove_all(ar);
}

/**
 * Gets an CC_Array element from the specified index and sets the out parameter to
 * its value. The specified index must be within the bounds of the array.
 *
 * @param[in] ar the array from which the element is being retrieved
 * @param[in] index the index of the array element
 * @param[out] out pointer to where the element is stored
 *
 * @return CC_OK if the element was found, or CC_ERR_OUT_OF_RANGE if the index
 * was out of range.
 */
enum cc_stat cc_array_get_at(const CC_Array *ar, size_t index, void **out) {
    if (index >= ar->size) {
        return CC_ERR_OUT_OF_RANGE;
    }

    *out = ar->buffer[index];
    return CC_OK;
}

/**
 * Gets the last element of the array or the element at the highest index
 * and sets the out parameter to its value.
 *
 * @param[in] ar the array whose last element is being returned
 * @param[out] out pointer to where the element is stored
 *
 * @return CC_OK if the element was found, or CC_ERR_VALUE_NOT_FOUND if the
 * CC_Array is empty.
 */
enum cc_stat cc_array_get_last(const CC_Array *ar, void **out) {
    if (ar->size == 0) {
        return CC_ERR_VALUE_NOT_FOUND;
    }

    return cc_array_get_at(ar, ar->size - 1, out);
}

/**
 * Returns the underlying array buffer.
 *
 * @note Any direct modification of the buffer may invalidate the CC_Array.
 *
 * @param[in] ar array whose underlying buffer is being returned
 *
 * @return array's internal buffer.
 */
const void *const *cc_array_get_buffer(CC_Array *ar) {
    return (const void *const *)ar->buffer;
}

/**
 * Gets the index of the specified element. The returned index is the index
 * of the first occurrence of the element starting from the beginning of the
 * CC_Array.
 *
 * @param[in] ar array being searched
 * @param[in] element the element whose index is being looked up
 * @param[out] index  pointer to where the index is stored
 *
 * @return CC_OK if the index was found, or CC_OUT_OF_RANGE if not.
 */
enum cc_stat cc_array_index_of(const CC_Array *ar, void *element, size_t *index) {
    size_t i;
    for (i = 0; i < ar->size; i++) {
        if (ar->buffer[i] == element) {
            *index = i;
            return CC_OK;
        }
    }
    return CC_ERR_OUT_OF_RANGE;
}

/**
 * Creates a subarray of the specified CC_Array, ranging from <code>b</code>
 * index (inclusive) to <code>e</code> index (inclusive). The range indices
 * must be within the bounds of the CC_Array, while the <code>e</code> index
 * must be greater or equal to the <code>b</code> index.
 *
 * @note The new CC_Array is allocated using the original CC_Array's allocators
 *       and it also inherits the configuration of the original CC_Array.
 *
 * @param[in] ar array from which the subarray is being created
 * @param[in] b the beginning index (inclusive) of the subarray that must be
 *              within the bounds of the array and must not exceed the
 *              the end index
 * @param[in] e the end index (inclusive) of the subarray that must be within
 *              the bounds of the array and must be greater or equal to the
 *              beginning index
 * @param[out] out pointer to where the new sublist is stored
 *
 * @return CC_OK if the subarray was successfully created, CC_ERR_INVALID_RANGE
 * if the specified index range is invalid, or CC_ERR_ALLOC if the memory allocation
 * for the new subarray failed.
 */
enum cc_stat cc_array_subarray(CC_Array *ar, size_t b, size_t e, CC_Array **out) {
    if (b > e || e >= ar->size) {
        return CC_ERR_INVALID_RANGE;
    }

    CC_Array *sub_ar = (CC_Array *)ar->mem_calloc(1, sizeof(CC_Array));

    if (!sub_ar) {
        return CC_ERR_ALLOC;
    }

    /* Try to allocate the buffer */
    if (!(sub_ar->buffer = (void **)ar->mem_alloc(ar->capacity * sizeof(void *)))) {
        ar->mem_free(sub_ar);
        return CC_ERR_ALLOC;
    }

    sub_ar->mem_alloc = ar->mem_alloc;
    sub_ar->mem_calloc = ar->mem_calloc;
    sub_ar->mem_free = ar->mem_free;
    sub_ar->size = e - b + 1;
    sub_ar->capacity = sub_ar->size;

    memcpy(sub_ar->buffer,
           &(ar->buffer[b]),
           sub_ar->size * sizeof(void *));

    *out = sub_ar;
    return CC_OK;
}

/**
 * Creates a shallow copy of the specified CC_Array. A shallow copy is a copy of
 * the CC_Array structure, but not the elements it holds.
 *
 * @note The new CC_Array is allocated using the original CC_Array's allocators
 *       and it also inherits the configuration of the original array.
 *
 * @param[in] ar the array to be copied
 * @param[out] out pointer to where the newly created copy is stored
 *
 * @return CC_OK if the copy was successfully created, or CC_ERR_ALLOC if the
 * memory allocation for the copy failed.
 */
enum cc_stat cc_array_copy_shallow(CC_Array *ar, CC_Array **out) {
    CC_Array *copy = (CC_Array *)ar->mem_alloc(sizeof(CC_Array));

    if (!copy) {
        return CC_ERR_ALLOC;
    }

    if (!(copy->buffer = (void **)ar->mem_calloc(ar->capacity, sizeof(void *)))) {
        ar->mem_free(copy);
        return CC_ERR_ALLOC;
    }
    copy->exp_factor = ar->exp_factor;
    copy->size = ar->size;
    copy->capacity = ar->capacity;
    copy->mem_alloc = ar->mem_alloc;
    copy->mem_calloc = ar->mem_calloc;
    copy->mem_free = ar->mem_free;

    memcpy(copy->buffer,
           ar->buffer,
           copy->size * sizeof(void *));

    *out = copy;
    return CC_OK;
}

/**
 * Creates a deep copy of the specified CC_Array. A deep copy is a copy of
 * both the CC_Array structure and the data it holds.
 *
 * @note The new CC_Array is allocated using the original CC_Array's allocators
 *       and it also inherits the configuration of the original CC_Array.
 *
 * @param[in] ar   array to be copied
 * @param[in] cp   the copy function that should return a pointer to the copy of
 *                 the data
 * @param[out] out pointer to where the newly created copy is stored
 *
 * @return CC_OK if the copy was successfully created, or CC_ERR_ALLOC if the
 * memory allocation for the copy failed.
 */
enum cc_stat cc_array_copy_deep(CC_Array *ar, void *(*cp)(void *), CC_Array **out) {
    CC_Array *copy = (CC_Array *)ar->mem_alloc(sizeof(CC_Array));

    if (!copy) {
        return CC_ERR_ALLOC;
    }

    if (!(copy->buffer = (void **)ar->mem_calloc(ar->capacity, sizeof(void *)))) {
        ar->mem_free(copy);
        return CC_ERR_ALLOC;
    }

    copy->exp_factor = ar->exp_factor;
    copy->size = ar->size;
    copy->capacity = ar->capacity;
    copy->mem_alloc = ar->mem_alloc;
    copy->mem_calloc = ar->mem_calloc;
    copy->mem_free = ar->mem_free;

    size_t i;
    for (i = 0; i < copy->size; i++) {
        copy->buffer[i] = cp(ar->buffer[i]);
    }

    *out = copy;

    return CC_OK;
}

/**
 * Filters the CC_Array by modifying it. It removes all elements that don't
 * return true on pred(element).
 *
 * @param[in] ar   array that is to be filtered
 * @param[in] pred predicate function which returns true if the element should
 *                 be kept in the CC_Array
 *
 * @return CC_OK if the CC_Array was filtered successfully, or CC_ERR_OUT_OF_RANGE
 * if the CC_Array is empty.
 */
enum cc_stat cc_array_filter_mut(CC_Array *ar, bool (*pred)(const void *)) {
    if (ar->size == 0) {
        return CC_ERR_OUT_OF_RANGE;
    }

    size_t rm = 0;
    size_t keep = 0;

    /* Look for clusters of non matching elements before moving
     * in order to minimize the number of memmoves */
    for (size_t i = ar->size - 1; i != ((size_t)-1); i--) {
        if (!pred(ar->buffer[i])) {
            rm++;
            continue;
        }
        if (rm > 0) {
            if (keep > 0) {
                size_t block_size = keep * sizeof(void *);
                memmove(&(ar->buffer[i + 1]),
                        &(ar->buffer[i + 1 + rm]),
                        block_size);
            }
            ar->size -= rm;
            rm = 0;
        }
        keep++;
    }
    /* Remove any remaining elements*/
    if (rm > 0) {
        size_t block_size = keep * sizeof(void *);
        memmove(&(ar->buffer[0]),
                &(ar->buffer[rm]),
                block_size);

        ar->size -= rm;
    }
    return CC_OK;
}

/**
 * Filters the CC_Array by creating a new CC_Array that contains all elements from the
 * original CC_Array that return true on pred(element) without modifying the original
 * CC_Array.
 *
 * @param[in] ar   array that is to be filtered
 * @param[in] pred predicate function which returns true if the element should
 *                 be kept in the filtered array
 * @param[out] out pointer to where the new filtered CC_Array is to be stored
 *
 * @return CC_OK if the CC_Array was filtered successfully, CC_ERR_OUT_OF_RANGE
 * if the CC_Array is empty, or CC_ERR_ALLOC if the memory allocation for the
 * new CC_Array failed.
 */
enum cc_stat cc_array_filter(CC_Array *ar, bool (*pred)(const void *), CC_Array **out) {
    if (ar->size == 0) {
        return CC_ERR_OUT_OF_RANGE;
    }

    CC_Array *filtered = (CC_Array *)ar->mem_alloc(sizeof(CC_Array));

    if (!filtered) {
        return CC_ERR_ALLOC;
    }

    if (!(filtered->buffer = (void **)ar->mem_calloc(ar->capacity, sizeof(void *)))) {
        ar->mem_free(filtered);
        return CC_ERR_ALLOC;
    }

    filtered->exp_factor = ar->exp_factor;
    filtered->size = 0;
    filtered->capacity = ar->capacity;
    filtered->mem_alloc = ar->mem_alloc;
    filtered->mem_calloc = ar->mem_calloc;
    filtered->mem_free = ar->mem_free;

    size_t f = 0;
    for (size_t i = 0; i < ar->size; i++) {
        if (pred(ar->buffer[i])) {
            filtered->buffer[f++] = ar->buffer[i];
            filtered->size++;
        }
    }
    *out = filtered;

    return CC_OK;
}

/**
 * Reverses the order of elements in the specified array.
 *
 * @param[in] ar array that is being reversed
 */
void cc_array_reverse(CC_Array *ar) {
    if (ar->size == 0) {
        return;
    }

    size_t i;
    size_t j;
    for (i = 0, j = ar->size - 1; i < ar->size / 2; i++, j--) {
        void *tmp = ar->buffer[i];
        ar->buffer[i] = ar->buffer[j];
        ar->buffer[j] = tmp;
    }
}

/**
 * Trims the array's capacity, in other words, it shrinks the capacity to match
 * the number of elements in the CC_Array, however the capacity will never shrink
 * below 1.
 *
 * @param[in] ar array whose capacity is being trimmed
 *
 * @return CC_OK if the capacity was trimmed successfully, or CC_ERR_ALLOC if
 * the reallocation failed.
 */
enum cc_stat cc_array_trim_capacity(CC_Array *ar) {
    if (ar->size == ar->capacity) {
        return CC_OK;
    }

    void **new_buff = (void **)ar->mem_calloc(ar->size, sizeof(void *));

    if (!new_buff) {
        return CC_ERR_ALLOC;
    }

    size_t size = ar->size < 1 ? 1 : ar->size;

    memcpy(new_buff, ar->buffer, size * sizeof(void *));
    ar->mem_free(ar->buffer);

    ar->buffer = new_buff;
    ar->capacity = ar->size;

    return CC_OK;
}

/**
 * Returns the number of occurrences of the element within the specified CC_Array.
 *
 * @param[in] ar array that is being searched
 * @param[in] element the element that is being searched for
 *
 * @return the number of occurrences of the element.
 */
size_t cc_array_contains(const CC_Array *ar, void *element) {
    size_t o = 0;
    size_t i;
    for (i = 0; i < ar->size; i++) {
        if (ar->buffer[i] == element) {
            o++;
        }
    }
    return o;
}

/**
 * Returns the number of occurrences of the value pointed to by <code>e</code>
 * within the specified CC_Array.
 *
 * @param[in] ar array that is being searched
 * @param[in] element the element that is being searched for
 * @param[in] cmp comparator function which returns 0 if the values passed to it are equal
 *
 * @return the number of occurrences of the value.
 */
size_t cc_array_contains_value(const CC_Array *ar, void *element, int (*cmp)(const void *, const void *)) {
    size_t o = 0;
    size_t i;
    for (i = 0; i < ar->size; i++) {
        if (cmp(element, ar->buffer[i]) == 0) {
            o++;
        }
    }
    return o;
}

/**
 * Returns the size of the specified CC_Array. The size of the array is the
 * number of elements contained within the CC_Array.
 *
 * @param[in] ar array whose size is being returned
 *
 * @return the the number of element within the CC_Array.
 */
size_t cc_array_size(const CC_Array *ar) {
    return ar->size;
}

/**
 * Returns the capacity of the specified CC_Array. The capacity of the CC_Array is
 * the maximum number of elements an CC_Array can hold before it has to be resized.
 *
 * @param[in] ar array whose capacity is being returned
 *
 * @return the capacity of the CC_Array.
 */
size_t cc_array_capacity(const CC_Array *ar) {
    return ar->capacity;
}

/**
 * Sorts the specified array.
 *
 * @note
 * Pointers passed to the comparator function will be pointers to the array
 * elements that are of type (void*) ie. void**. So an extra step of
 * dereferencing will be required before the data can be used for comparison:
 * eg. <code>my_type e = *(*((my_type**) ptr));</code>.
 *
 * @code
 * enum cc_stat mycmp(const void *e1, const void *e2) {
 *     MyType el1 = *(*((enum cc_stat**) e1));
 *     MyType el2 = *(*((enum cc_stat**) e2));
 *
 *     if (el1 < el2) return -1;
 *     if (el1 > el2) return 1;
 *     return 0;
 * }
 *
 * ...
 *
 * cc_array_sort(array, mycmp);
 * @endcode
 *
 * @param[in] ar  array to be sorted
 * @param[in] cmp the comparator function that must be of type <code>
 *                enum cc_stat cmp(const void e1*, const void e2*)</code> that
 *                returns < 0 if the first element goes before the second,
 *                0 if the elements are equal and > 0 if the second goes
 *                before the first
 */
void cc_array_sort(CC_Array *ar, int (*cmp)(const void *, const void *)) {
    qsort(ar->buffer, ar->size, sizeof(void *), cmp);
}

/**
 * Expands the CC_Array capacity. This might fail if the the new buffer
 * cannot be allocated. In case the expansion would overflow the index
 * range, a maximum capacity buffer is allocated instead. If the capacity
 * is already at the maximum capacity, no new buffer is allocated.
 *
 * @param[in] ar array whose capacity is being expanded
 *
 * @return CC_OK if the buffer was expanded successfully, CC_ERR_ALLOC if
 * the memory allocation for the new buffer failed, or CC_ERR_MAX_CAPACITY
 * if the array is already at maximum capacity.
 */
static enum cc_stat expand_array_capacity(CC_Array *ar) {
    if (ar->capacity == CC_MAX_ELEMENTS) {
        return CC_ERR_MAX_CAPACITY;
    }

    size_t new_capacity = (size_t)(ar->capacity * ar->exp_factor);

    /* As long as the capacity is greater that the expansion factor
     * at the point of overflow, this is check is valid. */
    if (new_capacity <= ar->capacity) {
        ar->capacity = CC_MAX_ELEMENTS;
    } else {
        ar->capacity = new_capacity;
    }

    void **new_buff = (void **)ar->mem_alloc(ar->capacity * sizeof(void *));

    if (!new_buff) {
        return CC_ERR_ALLOC;
    }

    memcpy(new_buff, ar->buffer, ar->size * sizeof(void *));

    ar->mem_free(ar->buffer);
    ar->buffer = new_buff;

    return CC_OK;
}

/**
 * Applies the function fn to each element of the CC_Array.
 *
 * @param[in] ar array on which this operation is performed
 * @param[in] fn operation function that is to be invoked on each CC_Array
 *               element
 */
void cc_array_map(CC_Array *ar, void (*fn)(void *e)) {
    size_t i;
    for (i = 0; i < ar->size; i++) {
        fn(ar->buffer[i]);
    }
}

/**
 * A fold/reduce function that collects all of the elements in the array
 * together. For example, if we have an array of [a,b,c...] the end result
 * will be (...((a+b)+c)+...).
 *
 * @param[in] ar the array on which this operation is performed
 * @param[in] fn the operation function that is to be invoked on each array
 *               element
 * @param[in] result the pointer which will collect the end result
 */
void cc_array_reduce(CC_Array *ar, void (*fn)(void *, void *, void *), void *result) {
    if (ar->size == 1) {
        fn(ar->buffer[0], NULL, result);
        return;
    }
    if (ar->size > 1) {
        fn(ar->buffer[0], ar->buffer[1], result);
    }

    for (size_t i = 2; i < ar->size; i++) {
        fn(result, ar->buffer[i], result);
    }
}

/**
 * Initializes the iterator.
 *
 * @param[in] iter the iterator that is being initialized
 * @param[in] ar the array to iterate over
 */
void cc_array_iter_init(CC_ArrayIter *iter, CC_Array *ar) {
    iter->ar = ar;
    iter->index = 0;
    iter->last_removed = false;
}

/**
 * Advances the iterator and sets the out parameter to the value of the
 * next element in the sequence.
 *
 * @param[in] iter the iterator that is being advanced
 * @param[out] out pointer to where the next element is set
 *
 * @return CC_OK if the iterator was advanced, or CC_ITER_END if the
 * end of the CC_Array has been reached.
 */
enum cc_stat cc_array_iter_next(CC_ArrayIter *iter, void **out) {
    if (iter->index >= iter->ar->size) {
        return CC_ITER_END;
    }

    *out = iter->ar->buffer[iter->index];

    iter->index++;
    iter->last_removed = false;

    return CC_OK;
}

/**
 * Removes the last returned element by <code>cc_array_iter_next()</code>
 * function without invalidating the iterator and optionally sets the out
 * parameter to the value of the removed element.
 *
 * @note This function should only ever be called after a call to <code>
 * cc_array_iter_next()</code>.

 * @param[in] iter the iterator on which this operation is being performed
 * @param[out] out pointer to where the removed element is stored, or NULL
 *                 if it is to be ignored
 *
 * @return CC_OK if the element was successfully removed, or
 * CC_ERR_VALUE_NOT_FOUND.
 */
enum cc_stat cc_array_iter_remove(CC_ArrayIter *iter, void **out) {
    enum cc_stat status = CC_ERR_VALUE_NOT_FOUND;

    if (!iter->last_removed) {
        status = cc_array_remove_at(iter->ar, iter->index - 1, out);
        if (status != CC_OK) {
            return status;
        }

        iter->last_removed = true;
        if (iter->index > 0) {
            iter->index--;
        }
    }
    return status;
}

/**
 * Removes the last returned element by <code>cc_array_iter_next()</code>
 * function without invalidating the iterator and optionally sets the out
 * parameter to the value of the removed element. The order of the array
 * is not preserved, the last element of the array is moved to the index
 * of the last returned element and the last element is removed.
 *
 * @note This function should only ever be called after a call to <code>
 * cc_array_iter_next()</code>.

 * @param[in] iter the iterator on which this operation is being performed
 * @param[out] out pointer to where the removed element is stored, or NULL
 *                 if it is to be ignored
 *
 * @return CC_OK if the element was successfully removed, or
 * CC_ERR_VALUE_NOT_FOUND.
 */
enum cc_stat cc_array_iter_remove_fast(CC_ArrayIter *iter, void **out) {
    enum cc_stat status = CC_ERR_VALUE_NOT_FOUND;

    if (!iter->last_removed) {
        status = cc_array_remove_fast_at(iter->ar, iter->index - 1, out);
        if (status != CC_OK) {
            return status;
        }

        iter->last_removed = true;
        if (iter->index > 0) {
            iter->index--;
        }
    }
    return status;
}

/**
 * Adds a new element to the CC_Array after the last returned element by
 * <code>cc_array_iter_next()</code> function without invalidating the
 * iterator.
 *
 * @note This function should only ever be called after a call to <code>
 * cc_array_iter_next()</code>.
 *
 * @param[in] iter the iterator on which this operation is being performed
 * @param[in] element the element being added
 *
 * @return CC_OK if the element was successfully added, CC_ERR_ALLOC if the
 * memory allocation for the new element failed, or CC_ERR_MAX_CAPACITY if
 * the array is already at maximum capacity.
 */
enum cc_stat cc_array_iter_add(CC_ArrayIter *iter, void *element) {
    return cc_array_add_at(iter->ar, element, iter->index++);
}

/**
 * Replaces the last returned element by <code>cc_array_iter_next()</code>
 * with the specified element and optionally sets the out parameter to
 * the value of the replaced element.
 *
 * @note This function should only ever be called after a call to <code>
 * cc_array_iter_next()</code>.
 *
 * @param[in] iter the iterator on which this operation is being performed
 * @param[in] element the replacement element
 * @param[out] out pointer to where the replaced element is stored, or NULL
 *                if it is to be ignored
 *
 * @return CC_OK if the element was replaced successfully, or
 * CC_ERR_OUT_OF_RANGE.
 */
enum cc_stat cc_array_iter_replace(CC_ArrayIter *iter, void *element, void **out) {
    return cc_array_replace_at(iter->ar, element, iter->index - 1, out);
}

/**
 * Returns the index of the last returned element by <code>cc_array_iter_next()
 * </code>.
 *
 * @note
 * This function should not be called before a call to <code>cc_array_iter_next()
 * </code>.
 *
 * @param[in] iter the iterator on which this operation is being performed
 *
 * @return the index.
 */
size_t cc_array_iter_index(CC_ArrayIter *iter) {
    return iter->index - 1;
}

/**
 * Initializes the zip iterator.
 *
 * @param[in] iter iterator that is being initialized
 * @param[in] ar1  first array
 * @param[in] ar2  second array
 */
void cc_array_zip_iter_init(CC_ArrayZipIter *iter, CC_Array *ar1, CC_Array *ar2) {
    iter->ar1 = ar1;
    iter->ar2 = ar2;
    iter->index = 0;
    iter->last_removed = false;
}

/**
 * Outputs the next element pair in the sequence and advances the iterator.
 *
 * @param[in]  iter iterator that is being advanced
 * @param[out] out1 output of the first array element
 * @param[out] out2 output of the second array element
 *
 * @return CC_OK if a next element pair is returned, or CC_ITER_END if the end of one
 * of the arrays has been reached.
 */
enum cc_stat cc_array_zip_iter_next(CC_ArrayZipIter *iter, void **out1, void **out2) {
    if (iter->index >= iter->ar1->size || iter->index >= iter->ar2->size) {
        return CC_ITER_END;
    }

    *out1 = iter->ar1->buffer[iter->index];
    *out2 = iter->ar2->buffer[iter->index];

    iter->index++;
    iter->last_removed = false;

    return CC_OK;
}

/**
 * Removes and outputs the last returned element pair by <code>cc_array_zip_iter_next()
 * </code> without invalidating the iterator.
 *
 * @param[in]  iter iterator on which this operation is being performed
 * @param[out] out1 output of the removed element from the first array
 * @param[out] out2 output of the removed element from the second array
 *
 * @return CC_OK if the element was successfully removed, CC_ERR_OUT_OF_RANGE if the
 * state of the iterator is invalid, or CC_ERR_VALUE_NOT_FOUND if the element was
 * already removed.
 */
enum cc_stat cc_array_zip_iter_remove(CC_ArrayZipIter *iter, void **out1, void **out2) {
    if ((iter->index - 1) >= iter->ar1->size || (iter->index - 1) >= iter->ar2->size) {
        return CC_ERR_OUT_OF_RANGE;
    }

    if (!iter->last_removed) {
        cc_array_remove_at(iter->ar1, iter->index - 1, out1);
        cc_array_remove_at(iter->ar2, iter->index - 1, out2);
        iter->last_removed = true;
        return CC_OK;
    }
    return CC_ERR_VALUE_NOT_FOUND;
}

/**
 * Adds a new element pair to the arrays after the last returned element pair by
 * <code>cc_array_zip_iter_next()</code> and immediately before an element pair
 * that would be returned by a subsequent call to <code>cc_array_zip_iter_next()</code>
 * without invalidating the iterator.
 *
 * @param[in] iter iterator on which this operation is being performed
 * @param[in] e1   element added to the first array
 * @param[in] e2   element added to the second array
 *
 * @return CC_OK if the element pair was successfully added to the arrays, or
 * CC_ERR_ALLOC if the memory allocation for the new elements failed.
 */
enum cc_stat cc_array_zip_iter_add(CC_ArrayZipIter *iter, void *e1, void *e2) {
    size_t index = iter->index++;
    CC_Array *ar1 = iter->ar1;
    CC_Array *ar2 = iter->ar2;

    /* Make sure both array buffers have room */
    if ((ar1->size == ar1->capacity && (expand_array_capacity(ar1) != CC_OK)) ||
        (ar2->size == ar2->capacity && (expand_array_capacity(ar2) != CC_OK))) {
        return CC_ERR_ALLOC;
    }

    cc_array_add_at(ar1, e1, index);
    cc_array_add_at(ar2, e2, index);

    return CC_OK;
}

/**
 * Replaces the last returned element pair by <code>cc_array_zip_iter_next()</code>
 * with the specified replacement element pair.
 *
 * @param[in] iter  iterator on which this operation is being performed
 * @param[in]  e1   first array's replacement element
 * @param[in]  e2   second array's replacement element
 * @param[out] out1 output of the replaced element from the first array
 * @param[out] out2 output of the replaced element from the second array
 *
 * @return CC_OK if the element was successfully replaced, or CC_ERR_OUT_OF_RANGE.
 */
enum cc_stat cc_array_zip_iter_replace(CC_ArrayZipIter *iter, void *e1, void *e2, void **out1, void **out2) {
    if ((iter->index - 1) >= iter->ar1->size || (iter->index - 1) >= iter->ar2->size) {
        return CC_ERR_OUT_OF_RANGE;
    }

    cc_array_replace_at(iter->ar1, e1, iter->index - 1, out1);
    cc_array_replace_at(iter->ar2, e2, iter->index - 1, out2);

    return CC_OK;
}

/**
 * Returns the index of the last returned element pair by <code>cc_array_zip_iter_next()</code>.
 *
 * @param[in] iter iterator on which this operation is being performed
 *
 * @return current iterator index.
 */
size_t cc_array_zip_iter_index(CC_ArrayZipIter *iter) {
    return iter->index - 1;
}

size_t cc_array_struct_size() {
    return sizeof(CC_Array);
}

#ifdef __cplusplus
}
#endif

#endif
