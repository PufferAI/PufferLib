# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False

'''Cythonized implementations of PufferLib's emulation functions

emulate is about 2x faster than Python. Nativize is only slightly faster.
'''

import numpy as np
cimport numpy as cnp

from pufferlib.spaces import Tuple, Dict, Discrete


def emulate(cnp.ndarray np_struct, object sample):
    cdef str k
    cdef int i

    if isinstance(sample, dict):
        for k, v in sample.items():
            emulate(np_struct[k], v)
    elif isinstance(sample, tuple):
        for i, v in enumerate(sample):
            emulate(np_struct[f'f{i}'], v)
    else:
        np_struct[()] = sample

cdef object _nativize(np_struct, object space):
    cdef str k
    cdef int i

    if isinstance(space, Discrete):
        return np_struct.item()
    elif isinstance(space, Tuple):
        return tuple(_nativize(np_struct[f'f{i}'], elem)
            for i, elem in enumerate(space))
    elif isinstance(space, Dict):
        return {k: _nativize(np_struct[k], value)
            for k, value in space.items()}
    else:
        return np_struct

def nativize(arr, object space, cnp.dtype struct_dtype):
    np_struct = np.asarray(arr).view(struct_dtype)[0]
    return _nativize(np_struct, space)
