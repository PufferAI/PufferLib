from pdb import set_trace as T
import numpy as np

import pufferlib.spaces
from pufferlib.emulation import flatten

def dtype_from_space(space):
    if isinstance(space, pufferlib.spaces.Tuple):
        dtype = []
        num_bytes = 0
        for i, elem in enumerate(space):
            dtype_ext, bytes_ext = dtype_from_space(elem)
            dtype.append((f'f{i}', dtype_ext))
            #dtype.append((dtype_ext,))
            num_bytes += bytes_ext
    elif isinstance(space, pufferlib.spaces.Dict):
        dtype = []
        num_bytes = 0
        for k, value in space.items():
            dtype_ext, bytes_ext = dtype_from_space(value)
            dtype.append((k, dtype_ext))
            num_bytes += bytes_ext
    else:
        dtype = (space.dtype, space.shape)
        num_bytes = space.dtype.itemsize * np.prod(space.shape)

    return dtype, num_bytes


def flat_dtype_from_space(space, name=None):
    dtype = []
    _flat_dtype_from_space(space, dtype, name)
    return dtype

def _flat_dtype_from_space(space, dtype, name=None):
    if isinstance(space, pufferlib.spaces.Tuple):
        for i, elem in enumerate(space):
            _flat_dtype_from_space(elem, dtype, name=f'f{i}')
            #_flat_dtype_from_space(elem, dtype, name=None)
    elif isinstance(space, pufferlib.spaces.Dict):
        for k, value in space.items():
            _flat_dtype_from_space(value, dtype, name=k)
    else:
        if name is not None:
            dtype.append((name, space.dtype, space.shape))
        else:
            dtype.append((space.dtype, space.shape))

    return dtype

def fill_with_sample(arr, sample):
    if isinstance(sample, dict):
        for k, v in sample.items():
            fill_with_sample(arr[k], v)
    elif isinstance(sample, tuple):
        for i, v in enumerate(sample):
            fill_with_sample(arr[f'f{i}'], v)
    else:
        arr[()] = sample


from gymnasium.spaces import Tuple, Dict, Box

test_space = Tuple([
    Dict({
        'a': Box(0, 1, shape=(2,)),
        'b': Box(0, 1, shape=(3,))
    }),
    Dict({
        'c': Box(0, 1, shape=(4,)),
    })
])

# Some notes:
# The flat version may be faster. It allows you to fill in a single
# numpy call, but you still have the precomputation step, although
# that one is with no copies. The main limit there is that you need
# unique dict keys everywhere to do it, since they are no longer
# unique when flat, even if they are in the structured version
# In either case, tuples need keys assigned for each element, which is
# the main limitation.

dtype, num_bytes = dtype_from_space(test_space)
dtype = np.dtype(dtype)
elem = np.zeros(1, dtype=dtype)
#flat_dtype = flat_dtype_from_space(test_space)
sample = test_space.sample()
fill_with_sample(elem, sample)
breakpoint()
#flat_sample = flatten(sample)
rec_array = np.rec.array(flat_sample, dtype=flat_dtype)
rec_array = rec_array.view(dtype)

'''
test_space = Dict({
    'a': Box(0, 1, shape=(3,)),
    'b': Dict({
        'c': Box(0, 1, shape=(4,)),
        'd': Box(0, 1, shape=(3,))
    }),
    'e': Box(0, 1, shape=(3,))
})
'''
 
breakpoint()
exit()

def mkdt(d):
    ll = []
    sz_bytes = 0
    for k,v in d.items():
        if isinstance(v,np.ndarray):
            ll.append((k,v.dtype))
            sz_bytes += v.nbytes
        else:
            l_ext, sz_ext = mkdt(v)
            ll.append((k,l_ext))
            sz_bytes += sz_ext
    return ll, sz_bytes

def mkdt_flat(d):
    dtype = []
    return _mkdt_flat(d, dtype)

def _mkdt_flat(d, dtype):
    for k,v in d.items():
        if isinstance(v,np.ndarray):
            dtype.append((k,v.dtype))
        else:
            _mkdt_flat(v, dtype)
    return dtype


arr1=np.arange(10).astype(np.float32)
arr2=np.arange(100.,110.).astype(np.uint8)
arr3=np.arange(200,210).astype(np.int32)
d={'a':arr1, 'b':{'b1':arr2, 'b2':{'c':arr3}}}
dt, sz_bytes = mkdt(d)

#A = np.zeros(sz_bytes, dtype=np.uint8)
flat = flatten(d)
flat_dtype = mkdt_flat(d)
rec_array = np.rec.array(flat, dtype=flat_dtype).view(dt)
