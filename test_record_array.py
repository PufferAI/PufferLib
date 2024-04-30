import gymnasium as gym
import numpy as np

# Create a custom Gym space using Dict, Tuple, and Box
space = gym.spaces.Dict({
    "position": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
    "velocity": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
    "description": gym.spaces.Tuple((
        #gym.spaces.Discrete(10),
        gym.spaces.Box(low=0, high=100, shape=(), dtype=np.int32),
        gym.spaces.Box(low=0, high=100, shape=(), dtype=np.int32)
    ))
})

space = gym.spaces.Dict({
    "position": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
})


# Define a function to create a dtype from the Gym space
def create_dtype_from_space(space):
    if isinstance(space, gym.spaces.Dict):
        dtype_fields = [(name, create_dtype_from_space(subspace)) for name, subspace in space.spaces.items()]
        return np.dtype(dtype_fields)
    elif isinstance(space, gym.spaces.Tuple):
        dtype_fields = [('field' + str(i), create_dtype_from_space(subspace)) for i, subspace in enumerate(space.spaces)]
        return np.dtype(dtype_fields)
    elif isinstance(space, gym.spaces.Box):
        return (space.dtype, space.shape)
    elif isinstance(space, gym.spaces.Discrete):
        return np.int64  # Assuming np.int64 for Discrete spaces

# Compute the dtype from the space
space_dtype = create_dtype_from_space(space)

sample = dict(space.sample())
breakpoint()
np.rec.array(sample, dtype=space_dtype)

# Function to sample from the space and convert to a structured numpy array
def sample_and_convert(space, dtype):
    sample = space.sample()
    flat_sample = {}
    def flatten(sample, name_prefix=""):
        for key, item in sample.items():
            full_key = name_prefix + key if name_prefix == "" else name_prefix + "_" + key
            if isinstance(item, dict):
                flatten(item, full_key)
            else:
                flat_sample[full_key] = item
    flatten(sample)
    return np.array(tuple(flat_sample.values()), dtype=dtype)

num_samples = 3
samples = [sample_and_convert(space, space_dtype) for _ in range(num_samples)]
print("Samples:", samples)

record_array = np.rec.array(samples)
print("Record Array:", record_array)

bytes_array = record_array.tobytes()
print("Bytes Array:", bytes_array)

record_array = np.rec.array(bytes_array, dtype=space_dtype)
print("Record Array from Bytes:", record_array)
