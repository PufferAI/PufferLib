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
    uint64_t state;
} BoxobanRandom;

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

static void boxoban_seed(BoxobanRandom* rng, uint64_t seed) {
    rng->state = seed ? seed : 0x9e3779b97f4a7c15ULL;
}

static uint64_t boxoban_next_u64(BoxobanRandom* rng) {
    uint64_t x = rng->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng->state = x;
    return x * 2685821657736338717ULL;
}

static uint32_t boxoban_randbelow(BoxobanRandom* rng, uint32_t n) {
    if (n == 0) {
        return 0;
    }

    uint64_t threshold = (uint64_t)(-(int64_t)n) % (uint64_t)n;
    for (;;) {
        uint64_t r = boxoban_next_u64(rng);
        if (r >= threshold) {
            return (uint32_t)(r % n);
        }
    }
}

static int boxoban_randint(BoxobanRandom* rng, int a, int b) {
    return a + (int)boxoban_randbelow(rng, (uint32_t)(b - a + 1));
}

static int boxoban_choice_index(BoxobanRandom* rng, int n) {
    return (int)boxoban_randbelow(rng, (uint32_t)n);
}

static int boxoban_sample_indices(BoxobanRandom* rng, int n, int k, int* out_indices) {
    int* pool = (int*)malloc((size_t)n * sizeof(int));
    if (pool == NULL) {
        return -1;
    }

    for (int i = 0; i < n; i++) {
        pool[i] = i;
    }

    for (int i = 0; i < k; i++) {
        int j = i + (int)boxoban_randbelow(rng, (uint32_t)(n - i));
        int tmp = pool[i];
        pool[i] = pool[j];
        pool[j] = tmp;
        out_indices[i] = pool[i];
    }

    free(pool);
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
    BoxobanRandom* rng,
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

        if (boxoban_sample_indices(rng, confined_count, num_boxes, sampled_idx) != 0) {
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

        if (boxoban_sample_indices(rng, box_candidate_count, num_boxes, sampled_idx) != 0) {
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

        BoxobanCell agent_cell = agent_candidates[boxoban_choice_index(rng, agent_candidate_count)];
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

    BoxobanRandom rng;
    boxoban_seed(&rng, seed);

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
            int box_count = num_boxes >= 1 ? num_boxes : boxoban_randint(&rng, min_boxes, max_boxes);
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
