# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=False

import numpy as np
cimport numpy as cnp

cpdef precompute_pathing(cnp.ndarray grid_np):
    cdef:
        int N = grid_np.shape[0]
        unsigned char[:, :] grid = grid_np
        unsigned char[:, :, :, :] paths = np.zeros((N, N, N, N), dtype=np.uint8) + 255
        int[:, :] buffer = np.zeros((8*N*N, 3), dtype=np.int32)
        int r, c

    for r in range(N):
        for c in range(N):
            bfs(grid, paths[r, c], buffer, r, c)

    return paths

cdef int bfs(unsigned char[:, :] grid, unsigned char[:, :] paths,
        int[:, :] buffer, int dest_r, int dest_c):
    cdef:
        int N = grid.shape[0]
        int start = 0
        int end = 1
        int atn, start_r, start_c

    if grid[dest_r, dest_c] == 1:
        return -1

    buffer[start, 0] = 0
    buffer[start, 1] = dest_r
    buffer[start, 2] = dest_c
    while start < end:
        atn = buffer[start, 0]
        start_r = buffer[start, 1]
        start_c = buffer[start, 2]
        start += 1

        if start_r < 0 or start_r >= N or start_c < 0 or start_c >= N:
            continue

        if paths[start_r, start_c] != 255:
            continue

        if grid[start_r, start_c] == 1:
            paths[start_r, start_c] = 8
            continue

        paths[start_r, start_c] = atn

        buffer[end, 0] = 0
        buffer[end, 1] = start_r - 1
        buffer[end, 2] = start_c
        end += 1

        buffer[end, 0] = 1
        buffer[end, 1] = start_r + 1
        buffer[end, 2] = start_c
        end += 1

        buffer[end, 0] = 2
        buffer[end, 1] = start_r
        buffer[end, 2] = start_c - 1
        end += 1

        buffer[end, 0] = 3
        buffer[end, 1] = start_r
        buffer[end, 2] = start_c + 1
        end += 1

        buffer[end, 0] = 4
        buffer[end, 1] = start_r - 1
        buffer[end, 2] = start_c + 1
        end += 1

        buffer[end, 0] = 5
        buffer[end, 1] = start_r + 1
        buffer[end, 2] = start_c + 1
        end += 1

        buffer[end, 0] = 6
        buffer[end, 1] = start_r + 1
        buffer[end, 2] = start_c - 1
        end += 1

        buffer[end, 0] = 7
        buffer[end, 1] = start_r - 1
        buffer[end, 2] = start_c - 1
        end += 1

    paths[dest_r, dest_c] = 8
    return end
