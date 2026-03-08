#ifndef PUFFERLIB_OCEAN_BOXOBAN_GENERATE_MAPS_H
#define PUFFERLIB_OCEAN_BOXOBAN_GENERATE_MAPS_H

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#define BOXOBAN_GEN_AGENT '@'
#define BOXOBAN_GEN_WALL '#'
#define BOXOBAN_GEN_BOX '$'
#define BOXOBAN_GEN_TARGET '.'
#define BOXOBAN_GEN_FLOOR ' '

typedef struct {
    int r;
    int c;
} BoxobanCell;

typedef struct {
    uint32_t mt[624];
    int index;
} BoxobanPyRandom;

static int boxoban_mkdir_p(const char* dir_path) {
    char tmp[1024];
    size_t len = strlen(dir_path);
    if (len >= sizeof(tmp)) {
        return -1;
    }

    memcpy(tmp, dir_path, len + 1);
    for (size_t i = 1; i < len; i++) {
        if (tmp[i] == '/') {
            tmp[i] = '\0';
            if (mkdir(tmp, 0777) != 0 && errno != EEXIST) {
                return -1;
            }
            tmp[i] = '/';
        }
    }
    if (mkdir(tmp, 0777) != 0 && errno != EEXIST) {
        return -1;
    }
    return 0;
}

static void boxoban_mt_seed_u32(BoxobanPyRandom* rng, uint32_t seed) {
    rng->mt[0] = seed;
    for (rng->index = 1; rng->index < 624; rng->index++) {
        rng->mt[rng->index] = 1812433253U * (rng->mt[rng->index - 1] ^ (rng->mt[rng->index - 1] >> 30)) + (uint32_t)rng->index;
    }
}

static void boxoban_mt_seed_by_array(BoxobanPyRandom* rng, const uint32_t* init_key, int key_len) {
    int i, j, k;
    boxoban_mt_seed_u32(rng, 19650218U);
    i = 1;
    j = 0;
    k = 624 > key_len ? 624 : key_len;
    for (; k > 0; k--) {
        rng->mt[i] = (rng->mt[i] ^ ((rng->mt[i - 1] ^ (rng->mt[i - 1] >> 30)) * 1664525U)) + init_key[j] + (uint32_t)j;
        i++;
        j++;
        if (i >= 624) {
            rng->mt[0] = rng->mt[623];
            i = 1;
        }
        if (j >= key_len) {
            j = 0;
        }
    }
    for (k = 623; k > 0; k--) {
        rng->mt[i] = (rng->mt[i] ^ ((rng->mt[i - 1] ^ (rng->mt[i - 1] >> 30)) * 1566083941U)) - (uint32_t)i;
        i++;
        if (i >= 624) {
            rng->mt[0] = rng->mt[623];
            i = 1;
        }
    }
    rng->mt[0] = 0x80000000U;
}

static void boxoban_py_seed(BoxobanPyRandom* rng, uint64_t seed) {
    uint32_t key[2];
    int key_len = 0;
    if (seed == 0) {
        key[0] = 0;
        key_len = 1;
    } else {
        while (seed != 0 && key_len < 2) {
            key[key_len++] = (uint32_t)(seed & 0xffffffffULL);
            seed >>= 32;
        }
    }
    boxoban_mt_seed_by_array(rng, key, key_len);
    rng->index = 624;
}

static uint32_t boxoban_mt_next_u32(BoxobanPyRandom* rng) {
    static const uint32_t mag01[2] = {0x0U, 0x9908b0dfU};
    uint32_t y;
    int kk;

    if (rng->index >= 624) {
        for (kk = 0; kk < 624 - 397; kk++) {
            y = (rng->mt[kk] & 0x80000000U) | (rng->mt[kk + 1] & 0x7fffffffU);
            rng->mt[kk] = rng->mt[kk + 397] ^ (y >> 1) ^ mag01[y & 0x1U];
        }
        for (; kk < 623; kk++) {
            y = (rng->mt[kk] & 0x80000000U) | (rng->mt[kk + 1] & 0x7fffffffU);
            rng->mt[kk] = rng->mt[kk + (397 - 624)] ^ (y >> 1) ^ mag01[y & 0x1U];
        }
        y = (rng->mt[623] & 0x80000000U) | (rng->mt[0] & 0x7fffffffU);
        rng->mt[623] = rng->mt[396] ^ (y >> 1) ^ mag01[y & 0x1U];
        rng->index = 0;
    }

    y = rng->mt[rng->index++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= (y >> 18);
    return y;
}

static int boxoban_bit_length_u32(uint32_t n) {
    int bits = 0;
    while (n != 0) {
        bits++;
        n >>= 1;
    }
    return bits;
}

static uint32_t boxoban_py_getrandbits(BoxobanPyRandom* rng, int k) {
    if (k <= 0) {
        return 0;
    }
    return boxoban_mt_next_u32(rng) >> (32 - k);
}

static uint32_t boxoban_py_randbelow(BoxobanPyRandom* rng, uint32_t n) {
    int k = boxoban_bit_length_u32(n);
    uint32_t r = boxoban_py_getrandbits(rng, k);
    while (r >= n) {
        r = boxoban_py_getrandbits(rng, k);
    }
    return r;
}

static int boxoban_py_randint(BoxobanPyRandom* rng, int a, int b) {
    return a + (int)boxoban_py_randbelow(rng, (uint32_t)(b - a + 1));
}

static int boxoban_py_choice_index(BoxobanPyRandom* rng, int n) {
    return (int)boxoban_py_randbelow(rng, (uint32_t)n);
}

static int boxoban_py_sample_indices(BoxobanPyRandom* rng, int n, int k, int* out_indices) {
    uint64_t setsize = 21;
    if (k > 5) {
        uint64_t target = (uint64_t)k * 3ULL;
        uint64_t pow4 = 1;
        while (pow4 < target) {
            pow4 *= 4ULL;
        }
        setsize += pow4;
    }

    if ((uint64_t)n <= setsize) {
        int* pool = (int*)malloc((size_t)n * sizeof(int));
        if (pool == NULL) {
            return -1;
        }
        for (int i = 0; i < n; i++) {
            pool[i] = i;
        }
        for (int i = 0; i < k; i++) {
            int j = (int)boxoban_py_randbelow(rng, (uint32_t)(n - i));
            out_indices[i] = pool[j];
            pool[j] = pool[n - i - 1];
        }
        free(pool);
        return 0;
    }

    uint8_t* selected = (uint8_t*)calloc((size_t)n, sizeof(uint8_t));
    if (selected == NULL) {
        return -1;
    }
    for (int i = 0; i < k; i++) {
        int j = (int)boxoban_py_randbelow(rng, (uint32_t)n);
        while (selected[j]) {
            j = (int)boxoban_py_randbelow(rng, (uint32_t)n);
        }
        selected[j] = 1;
        out_indices[i] = j;
    }
    free(selected);
    return 0;
}

static inline int boxoban_grid_idx(int size, int r, int c) {
    return r * size + c;
}

static int boxoban_is_inside(int size, int x, int y) {
    return x >= 0 && x < size && y >= 0 && y < size;
}

static int boxoban_is_pushable(const char* grid, int size, int x, int y) {
    static const int dirs[4][2] = {{1,0}, {-1,0}, {0,1}, {0,-1}};
    for (int d = 0; d < 4; d++) {
        int dx = dirs[d][0];
        int dy = dirs[d][1];
        int px = x - dx;
        int py = y - dy;
        int tx = x + dx;
        int ty = y + dy;
        if (!boxoban_is_inside(size, px, py) || !boxoban_is_inside(size, tx, ty)) {
            continue;
        }
        char pre = grid[boxoban_grid_idx(size, py, px)];
        char post = grid[boxoban_grid_idx(size, ty, tx)];
        if ((pre == BOXOBAN_GEN_FLOOR || pre == BOXOBAN_GEN_TARGET) &&
            (post == BOXOBAN_GEN_FLOOR || post == BOXOBAN_GEN_TARGET)) {
            return 1;
        }
    }
    return 0;
}

static void boxoban_build_border_grid(char* grid, int size) {
    for (int r = 0; r < size; r++) {
        for (int c = 0; c < size; c++) {
            grid[boxoban_grid_idx(size, r, c)] = BOXOBAN_GEN_FLOOR;
        }
    }
    for (int i = 0; i < size; i++) {
        grid[boxoban_grid_idx(size, 0, i)] = BOXOBAN_GEN_WALL;
        grid[boxoban_grid_idx(size, size - 1, i)] = BOXOBAN_GEN_WALL;
        grid[boxoban_grid_idx(size, i, 0)] = BOXOBAN_GEN_WALL;
        grid[boxoban_grid_idx(size, i, size - 1)] = BOXOBAN_GEN_WALL;
    }
}

static int boxoban_build_cells(int size, int margin, BoxobanCell* out_cells) {
    int count = 0;
    int start = 1 + margin;
    int end = size - 1 - margin;
    for (int r = start; r < end; r++) {
        for (int c = start; c < end; c++) {
            out_cells[count].r = r;
            out_cells[count].c = c;
            count++;
        }
    }
    return count;
}

static int boxoban_make_puzzle(
    int size,
    BoxobanPyRandom* rng,
    int num_boxes,
    int max_attempts,
    const BoxobanCell* agent_choices,
    int agent_count,
    const BoxobanCell* confined,
    int confined_count,
    int interior_count,
    char* grid
) {
    if (num_boxes < 1) {
        fprintf(stderr, "num_boxes must be at least 1\n");
        return -1;
    }

    int needed = num_boxes * 2 + 1;
    if (needed > interior_count) {
        fprintf(stderr, "Grid interior only has %d cells, cannot place %d objects\n", interior_count, needed);
        return -1;
    }

    BoxobanCell* box_candidates = (BoxobanCell*)malloc((size_t)confined_count * sizeof(BoxobanCell));
    BoxobanCell* box_positions = (BoxobanCell*)malloc((size_t)num_boxes * sizeof(BoxobanCell));
    BoxobanCell* agent_candidates = (BoxobanCell*)malloc((size_t)agent_count * sizeof(BoxobanCell));
    int* sampled_idx = (int*)malloc((size_t)num_boxes * sizeof(int));
    uint8_t* occupied = (uint8_t*)calloc((size_t)size * (size_t)size, sizeof(uint8_t));
    if (box_candidates == NULL || box_positions == NULL || agent_candidates == NULL || sampled_idx == NULL || occupied == NULL) {
        free(box_candidates);
        free(box_positions);
        free(agent_candidates);
        free(sampled_idx);
        free(occupied);
        return -1;
    }

    for (int attempt = 0; attempt < max_attempts; attempt++) {
        boxoban_build_border_grid(grid, size);
        memset(occupied, 0, (size_t)size * (size_t)size);

        if (boxoban_py_sample_indices(rng, confined_count, num_boxes, sampled_idx) != 0) {
            free(box_candidates);
            free(box_positions);
            free(agent_candidates);
            free(sampled_idx);
            free(occupied);
            return -1;
        }

        for (int i = 0; i < num_boxes; i++) {
            BoxobanCell cell = confined[sampled_idx[i]];
            grid[boxoban_grid_idx(size, cell.r, cell.c)] = BOXOBAN_GEN_TARGET;
            occupied[boxoban_grid_idx(size, cell.r, cell.c)] = 1;
        }

        int box_candidate_count = 0;
        for (int i = 0; i < confined_count; i++) {
            BoxobanCell cell = confined[i];
            if (!occupied[boxoban_grid_idx(size, cell.r, cell.c)]) {
                box_candidates[box_candidate_count++] = cell;
            }
        }
        if (box_candidate_count < num_boxes) {
            continue;
        }

        if (boxoban_py_sample_indices(rng, box_candidate_count, num_boxes, sampled_idx) != 0) {
            free(box_candidates);
            free(box_positions);
            free(agent_candidates);
            free(sampled_idx);
            free(occupied);
            return -1;
        }
        for (int i = 0; i < num_boxes; i++) {
            BoxobanCell cell = box_candidates[sampled_idx[i]];
            box_positions[i] = cell;
            grid[boxoban_grid_idx(size, cell.r, cell.c)] = BOXOBAN_GEN_BOX;
            occupied[boxoban_grid_idx(size, cell.r, cell.c)] = 1;
        }

        int agent_candidate_count = 0;
        for (int i = 0; i < agent_count; i++) {
            BoxobanCell cell = agent_choices[i];
            if (!occupied[boxoban_grid_idx(size, cell.r, cell.c)]) {
                agent_candidates[agent_candidate_count++] = cell;
            }
        }
        if (agent_candidate_count == 0) {
            continue;
        }

        BoxobanCell agent_cell = agent_candidates[boxoban_py_choice_index(rng, agent_candidate_count)];
        grid[boxoban_grid_idx(size, agent_cell.r, agent_cell.c)] = BOXOBAN_GEN_AGENT;

        int all_pushable = 1;
        for (int i = 0; i < num_boxes; i++) {
            BoxobanCell cell = box_positions[i];
            if (!boxoban_is_pushable(grid, size, cell.c, cell.r)) {
                all_pushable = 0;
                break;
            }
        }

        if (all_pushable) {
            free(box_candidates);
            free(box_positions);
            free(agent_candidates);
            free(sampled_idx);
            free(occupied);
            return 0;
        }
    }

    free(box_candidates);
    free(box_positions);
    free(agent_candidates);
    free(sampled_idx);
    free(occupied);
    fprintf(stderr, "Failed to sample a solvable puzzle after many attempts\n");
    return -1;
}

static int boxoban_generate_maps(
    const char* output_dir,
    int num_files,
    int puzzles_per_file,
    int size,
    int num_boxes,
    int min_boxes,
    int max_boxes,
    uint64_t seed
) {
    if (boxoban_mkdir_p(output_dir) != 0) {
        return -1;
    }

    BoxobanPyRandom rng;
    boxoban_py_seed(&rng, seed);

    int max_cells = (size - 2) * (size - 2);
    BoxobanCell* agent_choices = (BoxobanCell*)malloc((size_t)max_cells * sizeof(BoxobanCell));
    BoxobanCell* confined = (BoxobanCell*)malloc((size_t)max_cells * sizeof(BoxobanCell));
    char* grid = (char*)malloc((size_t)size * (size_t)size);
    if (agent_choices == NULL || confined == NULL || grid == NULL) {
        free(agent_choices);
        free(confined);
        free(grid);
        return -1;
    }

    int interior_count = (size - 2) * (size - 2);
    int agent_count = boxoban_build_cells(size, 0, agent_choices);
    int confined_count = boxoban_build_cells(size, 1, confined);

    for (int file_idx = 0; file_idx < num_files; file_idx++) {
        char out_path[1200];
        snprintf(out_path, sizeof(out_path), "%s/%03d.txt", output_dir, file_idx);
        FILE* out = fopen(out_path, "w");
        if (out == NULL) {
            free(agent_choices);
            free(confined);
            free(grid);
            return -1;
        }

        for (int puzzle_idx = 0; puzzle_idx < puzzles_per_file; puzzle_idx++) {
            int box_count = num_boxes >= 1 ? num_boxes : boxoban_py_randint(&rng, min_boxes, max_boxes);
            if (boxoban_make_puzzle(
                    size, &rng, box_count, 200, agent_choices, agent_count, confined, confined_count, interior_count, grid) != 0) {
                fclose(out);
                free(agent_choices);
                free(confined);
                free(grid);
                return -1;
            }

            fprintf(out, "; %d\n", puzzle_idx);
            for (int r = 0; r < size; r++) {
                fwrite(&grid[boxoban_grid_idx(size, r, 0)], 1, (size_t)size, out);
                fputc('\n', out);
            }
            fputc('\n', out);
        }

        fclose(out);
    }

    free(agent_choices);
    free(confined);
    free(grid);
    return 0;
}

static int boxoban_generate_easy_maps(const char* output_dir, uint64_t seed) {
    return boxoban_generate_maps(output_dir, 300, 1000, 10, -1, 1, 4, seed);
}

static int boxoban_generate_basic_maps(const char* output_dir, uint64_t seed) {
    return boxoban_generate_maps(output_dir, 300, 1000, 10, 1, 1, 4, seed);
}

#endif
