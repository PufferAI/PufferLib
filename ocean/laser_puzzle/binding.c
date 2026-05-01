#include "laser_puzzle.h"

#define OBS_SIZE (INIT_ROWS * INIT_COLS)
#define NUM_ATNS 1
#define ACT_SIZES {NUM_ACTIONS}
#define OBS_TENSOR_T ByteTensor

#define Env LaserPuzzle
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    // kwargs are passed in py the config .ini file, can set them here, will ignore for now
    (void)kwargs;
    env->num_agents = 1;
    env->ROWS = INIT_ROWS;
    env->COLS = INIT_COLS;
    env->max_steps = NUM_ACTIONS;
    env->owns_buffers = 0;

    // Only allocate env-owned state here. vecenv owns observations/actions/rewards/terminals and allocates it in the big buffer
    env->board = (Cell*)calloc(env->ROWS * env->COLS, sizeof(Cell));
    load_laser_puzzle_levels(env, LASER_PUZZLE_LEVELS_PATH);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
