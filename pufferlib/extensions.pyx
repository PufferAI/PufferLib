#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False

DICT = 0
LIST = 1
TUPLE = 2
VALUE = 4


cdef void flatten_helper(object d, list flat_data):
    cdef dict d_dict
    cdef list d_list
    cdef tuple d_tuple

    if isinstance(d, dict):
        d_dict = <dict> d
        for key, value in sorted(d_dict.items()):
            flatten_helper(value, flat_data)
    elif isinstance(d, list):
        d_list = <list> d
        for item in d_list.items:
            flatten_helper(item, flat_data)
    elif isinstance(d, tuple):
        d_tuple = <tuple> d
        for item in d_tuple:
            flatten_helper(item, flat_data)
    else:
        flat_data.append(d)

def flatten(data):
    cdef list flat_data = []
    flatten_helper(data, flat_data)
    return flat_data

cdef unflatten_helper(list flat_data, list structure, int struct_idx, int data_idx):
    cdef int n, token
    cdef object key, value
    cdef dict result_dict
    cdef list result_list

    token = <int> structure[struct_idx]
    struct_idx += 1

    if token == DICT:
        n = <int> structure[struct_idx]
        struct_idx += 1
        result_dict = {}
        for _ in range(n):
            key = structure[struct_idx]
            result_dict[key], struct_idx, data_idx = unflatten_helper(
                flat_data, structure, struct_idx + 1, data_idx)
        return result_dict, struct_idx, data_idx
    elif token == LIST:
        n = <int> structure[struct_idx]
        struct_idx += 1
        result_list = [None] * n
        for i in range(n):
            result_list[i], struct_idx, data_idx = unflatten_helper(
                flat_data, structure, struct_idx, data_idx)
        return result_list, struct_idx, data_idx
    elif token == TUPLE:
        n = <int> structure[struct_idx]
        struct_idx += 1
        result_list = [None] * n
        for i in range(n):
            result_list[i], struct_idx, data_idx = unflatten_helper(
                flat_data, structure, struct_idx, data_idx)
        return tuple(result_list), struct_idx, data_idx
    else:
        return flat_data[data_idx], struct_idx, data_idx + 1

def unflatten(list flat_data, list structure):
    return unflatten_helper(flat_data, structure, 0, 0)[0]
