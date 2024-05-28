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


def emulate(cnp.ndarray arr, object sample):
    cdef str k
    cdef int i

    if isinstance(sample, dict):
        for k, v in sample.items():
            emulate(arr[k], v)
    elif isinstance(sample, tuple):
        for i, v in enumerate(sample):
            emulate(arr[f'f{i}'], v)
    else:
        arr[()] = sample

cdef _nativize(object sample, object space):
    cdef str k
    cdef int i

    if isinstance(space, Discrete):
        return sample.item()
    elif isinstance(space, Tuple):
        return tuple(_nativize(sample[f'f{i}'], elem)
            for i, elem in enumerate(space))
    elif isinstance(space, Dict):
        return {k: _nativize(sample[k], value)
            for k, value in space.items()}
    else:
        return sample

def nativize(sample, object sample_space, cnp.dtype emulated_dtype):
    sample = np.asarray(sample).view(emulated_dtype)[0]
    return _nativize(sample, sample_space)
