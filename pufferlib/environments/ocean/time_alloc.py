import numpy as np
import timeit

# Time np.zeros(2, 5) for 100000 iterations
time_zeros = timeit.timeit('np.zeros((2, 5))', setup='import numpy as np', number=100000)

# Pre-allocate the array
preallocated_array = np.zeros((2, 5))

# Time setting the pre-allocated array to zero for 100000 iterations
time_preallocated = timeit.timeit('preallocated_array[:] = 0', setup='import numpy as np; preallocated_array = np.zeros((2, 5))', number=100000)

print(f"Time for np.zeros(2, 5) over 100000 iterations: {time_zeros} seconds")
print(f"Time for preallocated *= 0 over 100000 iterations: {time_preallocated} seconds")

