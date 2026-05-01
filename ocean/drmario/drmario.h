#include <stdlib.h>
#include <string.h>
#include "raylib.h"
#include <stdbool.h>
#include <time.h>

#define SQUARE_SIZE 32
#define TICKS_PER_FALL 3

#define SCORE_SOFT_DROP 0.0f
#define SCORE_HARD_DROP 0.0f
#define SCORE_ROTATE 0.0f
#define SCORE_KILL_VIRUS 1000.0f
#define SCORE_PLACE_NEXT_TO_SAME_COLOR 10.0f
#define SCORE_NO_LINE_CLEARS -10.0f
#define SCORE_CLEAR_LINE 500.0f

#define REWARD_SOFT_DROP 0.0f
#define REWARD_HARD_DROP 0.0f
#define REWARD_ROTATE 0.0f
#define REWARD_KILL_VIRUS 1.0f
#define REWARD_PLACE_NEXT_TO_SAME_COLOR 0.01f
#define REWARD_NO_LINE_CLEARS -0.01f
#define REWARD_CLEAR_LINE 0.5f
#define REWARD_HEIGHT -0.02f

#define ROTATION_0 0
#define ROTATION_90 1
#define ROTATION_180 2
#define ROTATION_270 3

#define ACTION_NO_OP 0
#define ACTION_LEFT 1
#define ACTION_RIGHT 2
#define ACTION_DOWN 3
#define ACTION_ROTATE_LEFT 4
#define ACTION_ROTATE_RIGHT 5
#define ACTION_DROP 6

#define N_SCALAR_OBS 12
#define N_OBS_PLANES 3

// Required struct. Only use floats!
typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported in binding.c
    float viruses_cleared;

    float n; // Required as the last field
} Log;

// Required that you have some struct for your env
typedef struct {
    int total_rows;
    int total_columns;
} Client;

typedef struct {
    Client *client;
    Log log;

    float *observations;
    float *actions;
    float *rewards;
    float *terminals;
    int dim_obs;

    int num_agents;

    int n_rows;
    int n_cols;
    int *grid;

    int cap_color_a;
    int cap_color_b;
    int cap_orient;
    int cap_row_1;
    int cap_col_1;
    int cap_row_2;
    int cap_col_2;

    bool cap_colliding_left;
    bool cap_colliding_right;
    bool cap_colliding_down; 
    bool cap_colliding_up;

    float cap_colliding_color_hor_1;
    float cap_colliding_color_hor_2;
    float cap_colliding_color_ver_1;
    float cap_colliding_color_ver_2;

    int tick;
    int tick_fall;
    int ticks_per_fall;

    int score;
    int stage;

    int viruses_remaining;
    int n_init_viruses;

    float episode_return;
    int viruses_cleared;

    int viruses_cleared_step;
    int lines_cleared_step;

    int atn_count_soft_drop;
    int atn_count_hard_drop;
    int atn_count_rotate;

    unsigned int rng;
} DrMario;


void c_init(DrMario *env) {
    env->grid = (int*)calloc(env->n_rows*env->n_cols, sizeof(int));
    if (env->grid == NULL) {
        exit(1);
    }
}

void allocate(DrMario *env) {
    c_init(env);
    env->dim_obs = env->n_rows*env->n_cols*N_OBS_PLANES + N_SCALAR_OBS; 
    env->observations = (float *)calloc(env->dim_obs, sizeof(float));
    if (env->observations == NULL) {
        exit(1);
    }
    env->actions = (float *)calloc(1, sizeof(float));
    if (env->actions == NULL) {
        exit(1);
    }
    env->rewards = (float *)calloc(1, sizeof(float));
    if (env->rewards == NULL) {
        exit(1);
    }
    env->terminals = (float *)calloc(1, sizeof(float));
    if (env->terminals == NULL) {
        exit(1);
    }
}

void c_close(DrMario *env) {
    free(env->grid);
    if (IsWindowReady()) {
       CloseWindow();
    }
}

void free_allocated(DrMario *env) {
    free(env->actions);
    free(env->observations);
    free(env->terminals);
    free(env->rewards);
    c_close(env);
}


void add_log(DrMario *env) {
    env->log.perf += env->viruses_cleared / (float)env->n_init_viruses;
    env->log.score += env->score;
    env->log.episode_length += env->tick;
    env->log.episode_return += env->episode_return;
    env->log.viruses_cleared += env->viruses_cleared;
    env->log.n++;
}


void compute_observations(DrMario *env) {
    int cells = env->n_rows * env->n_cols;
    
    float* plane_occupied = env->observations;
    float* plane_viruses = env->observations + cells;
    float* plane_colors = env->observations + 2*cells;

    for (int i = 0; i < cells; i++) {
        int cell = env->grid[i];
        plane_occupied[i] = cell != 0 ? 1.0f : 0.0f;
        plane_viruses[i] = cell < 0 ? 1.0f : 0.0f;
        plane_colors[i] = cell != 0 ? abs(cell) / 3.0f : 0.0f;
    }

    int r1 = env->cap_row_1, c1 = env->cap_col_1;
    int r2 = env->cap_row_2, c2 = env->cap_col_2;
    if (r1 >= 0 && r1 < env->n_rows && c1 >= 0 && c1 < env->n_cols) {
        int i = r1*env->n_cols + c1;
        plane_occupied[i] = 1.0f;
        plane_viruses[i] = 0.0f;
        plane_colors[i] = env->cap_color_a / 3.0f;
    }
    if (r2 >= 0 && r2 < env->n_rows && c2 >= 0 && c2 < env->n_cols) {
        int i = r2*env->n_cols + c2;
        plane_occupied[i] = 1.0f;
        plane_viruses[i] = 0.0f;
        plane_colors[i] = env->cap_color_b / 3.0f;
    }

    int off = cells * N_OBS_PLANES;
    float safe_r1 = (r1 < 0) ? 0.0f : r1 / (float)(env->n_rows - 1);
    float safe_r2 = (r2 < 0) ? 0.0f : r2 / (float)(env->n_rows - 1);
    env->observations[off + 0] = env->cap_color_a / 3.0f;
    env->observations[off + 1] = env->cap_color_b / 3.0f;
    env->observations[off + 2] = env->cap_orient / 3.0f;
    env->observations[off + 3] = safe_r1;
    env->observations[off + 4] = c1 / (float)(env->n_cols - 1);
    env->observations[off + 5] = safe_r2;
    env->observations[off + 6] = c2 / (float)(env->n_cols - 1);
    env->observations[off + 7] = env->viruses_remaining / (float)env->n_init_viruses;
    env->observations[off + 8] = env->viruses_cleared_step / (float)env->n_init_viruses;
    env->observations[off + 9] = env->lines_cleared_step / 4.0f;
    env->observations[off + 10] = env->score / 10000.0f;
    env->observations[off + 11] = env->tick / 2000.0f;
}

void place_viruses(DrMario *env) {
    env->viruses_remaining = 0;
    int placed = 0;
    int attempts = 0;
    
    while (placed < env->n_init_viruses && attempts < 1000) {
        attempts++;
        int r = (rand_r(&env->rng) % 8) + 8;
        int c = rand_r(&env->rng) % env->n_cols;
        int idx = r*env->n_cols + c;
        
        if (env->grid[idx] != 0) {
            continue;
        }
        
        int color = (rand_r(&env->rng) % 3) + 1;
        env->grid[idx] = -color;
        placed++;
        env->viruses_remaining++;
    }
}

void spawn_capsule(DrMario *env) {
    env->cap_color_a = rand_r(&env->rng) % 3 + 1;
    env->cap_color_b = rand_r(&env->rng) % 3 + 1;
    env->cap_orient = ROTATION_0;
    env->cap_row_1 = -1;
    env->cap_col_1 = env->n_cols / 2;
    env->cap_row_2 = env->cap_row_1;
    env->cap_col_2 = env->cap_col_1 + 1;
    env->tick_fall = 0;
}

void c_reset(DrMario *env) {
    memset(env->grid, 0, env->n_rows*env->n_cols*sizeof(int));
    env->score = 0;
    env->tick = 0;
    env->tick_fall = 0;

    env->ticks_per_fall = TICKS_PER_FALL;
    env->viruses_remaining = env->n_init_viruses;

    env->episode_return = 0;
    env->viruses_cleared = 0;
    env->atn_count_soft_drop = 0;
    env->atn_count_hard_drop = 0;
    env->atn_count_rotate = 0;
    place_viruses(env);
    spawn_capsule(env);
    compute_observations(env);
}

void get_collisions(DrMario* env) {
    env->cap_colliding_left = false;
    env->cap_colliding_right = false;
    env->cap_colliding_down = false;
    env->cap_colliding_up = false;

    if (env->cap_row_1 < 0 || env->cap_row_2 < 0) {
        return;
    }

    int below1 = (env->cap_row_1+1)*env->n_cols + env->cap_col_1;
    int below2 = (env->cap_row_2+1)*env->n_cols + env->cap_col_2;
    bool blocked_down = env->grid[below1] != 0 || env->grid[below2] != 0;
    bool at_bottom = env->cap_row_1 == env->n_rows-1 || env->cap_row_2 == env->n_rows-1;
    if (blocked_down || at_bottom) {
        env->cap_colliding_down = true;
    }

    int above1 = (env->cap_row_1-1)*env->n_cols + env->cap_col_1;
    int above2 = (env->cap_row_2-1)*env->n_cols + env->cap_col_2;
    bool blocked_up = env->grid[above1] != 0 || env->grid[above2] != 0;
    if (blocked_up) {
        env->cap_colliding_up = true;
    }

    int right1 = env->cap_row_1*env->n_cols + env->cap_col_1 + 1;
    int right2 = env->cap_row_2*env->n_cols + env->cap_col_2 + 1;
    bool blocked_right = env->grid[right1] != 0 || env->grid[right2] != 0;
    bool at_right_wall = env->cap_col_1 == env->n_cols-1 || env->cap_col_2 == env->n_cols-1;
    if (blocked_right || at_right_wall) {
        env->cap_colliding_right = true;
    }

    int left1 = env->cap_row_1*env->n_cols + env->cap_col_1 - 1;
    int left2 = env->cap_row_2*env->n_cols + env->cap_col_2 - 1;
    bool blocked_left = env->grid[left1] != 0 || env->grid[left2] != 0;
    bool at_left_wall = env->cap_col_1 == 0 || env->cap_col_2 == 0;
    if (blocked_left || at_left_wall) {
        env->cap_colliding_left = true;
    }
}

void get_color_collisions(DrMario* env) {
    env->cap_colliding_color_hor_1 = 0.0f;
    env->cap_colliding_color_ver_1 = 0.0f;
    env->cap_colliding_color_hor_2 = 0.0f;
    env->cap_colliding_color_ver_2 = 0.0f;

    if (env->cap_row_1 < 0 || env->cap_row_2 < 0) {
        return;
    }

    int cap_color_1 = abs(env->cap_color_a);
    int cap_color_2 = abs(env->cap_color_b);
    int cols = env->n_cols;

    if (env->cap_row_1+1 != env->cap_row_2) {
        int i = 1;
        int color_up = abs(env->grid[(env->cap_row_1+i)*cols + env->cap_col_1]);
        while (color_up == cap_color_1 && i < 100) {
            env->cap_colliding_color_ver_1++;
            i++;
            color_up = abs(env->grid[(env->cap_row_1+i)*cols + env->cap_col_1]);
        }
    }

    if (env->cap_row_1 >= 1 && env->cap_row_1-1 != env->cap_row_2) {
        int i = 1;
        int color_down = abs(env->grid[(env->cap_row_1-i)*cols + env->cap_col_1]);
        while (color_down == cap_color_1 && i <= 100) {
            env->cap_colliding_color_ver_1++;
            i++;
            color_down = abs(env->grid[(env->cap_row_1-i)*cols + env->cap_col_1]);
        }
    }

    if (env->cap_col_1+1 != env->cap_col_2) {
        int i = 1;
        int color_right = abs(env->grid[env->cap_row_1*cols + env->cap_col_1+i]);
        while (color_right == cap_color_1 && i <= 100) {
            env->cap_colliding_color_hor_1++;
            i++;
            color_right = abs(env->grid[env->cap_row_1*cols + env->cap_col_1+i]);
        }
    }

    if (env->cap_col_1 >= 1 && env->cap_col_1-1 != env->cap_col_2) {
        int i = 1;
        int color_left = abs(env->grid[env->cap_row_1*cols + env->cap_col_1-i]);
        while (color_left == cap_color_1 && i <= 100) {
            env->cap_colliding_color_hor_1++;
            i++;
            color_left = abs(env->grid[env->cap_row_1*cols + env->cap_col_1-i]);
        }
    }

    if (env->cap_row_2+1 != env->cap_row_1) {
        int i = 1;
        int color_up = abs(env->grid[(env->cap_row_2+i)*cols + env->cap_col_2]);
        while (color_up == cap_color_2 && i < 100) {
            env->cap_colliding_color_ver_2++;
            i++;
            color_up = abs(env->grid[(env->cap_row_2+i)*cols + env->cap_col_2]);
        }
    }

    if (env->cap_row_2 >= 1 && env->cap_row_2-1 != env->cap_row_1) {
        int i = 1;
        int color_down = abs(env->grid[(env->cap_row_2-i)*cols + env->cap_col_2]);
        while (color_down == cap_color_2 && i <= 100) {
            env->cap_colliding_color_ver_2++;
            i++;
            color_down = abs(env->grid[(env->cap_row_2-i)*cols + env->cap_col_2]);
        }
    }

    if (env->cap_col_2+1 != env->cap_col_1) {
        int i = 1;
        int color_right = abs(env->grid[env->cap_row_2*cols + env->cap_col_2+i]);
        while (color_right == cap_color_2 && i <= 100) {
            env->cap_colliding_color_hor_2++;
            i++;
            color_right = abs(env->grid[env->cap_row_2*cols + env->cap_col_2+i]);
        }
    }

    if (env->cap_col_2 >= 1 && env->cap_col_2-1 != env->cap_col_1) {
        int i = 1;
        int color_left = abs(env->grid[env->cap_row_2*cols + env->cap_col_2-i]);
        while (color_left == cap_color_2 && i <= 100) {
            env->cap_colliding_color_hor_2++;
            i++;
            color_left = abs(env->grid[env->cap_row_2*cols + env->cap_col_2-i]);
        }
    }
}

void rotate_cap(DrMario* env) {
    int old_orient = env->cap_orient;
    int old_cap_row_2 = env->cap_row_2;
    int old_cap_col_2 = env->cap_col_2;

    if (env->actions[0] == ACTION_ROTATE_LEFT) {
        env->cap_orient = (env->cap_orient + 1) % 4;
    } else if (env->actions[0] == ACTION_ROTATE_RIGHT) {
        env->cap_orient = (env->cap_orient + 3) % 4;
    } else {
        return;
    }

    env->cap_row_2 = env->cap_row_1;
    if (env->cap_orient == ROTATION_90) {
        env->cap_row_2 -= 1;
    } else if (env->cap_orient == ROTATION_270) {
        env->cap_row_2 += 1;
    }

    env->cap_col_2 = env->cap_col_1;
    if (env->cap_orient == ROTATION_0) {
        env->cap_col_2 += 1;
    } else if (env->cap_orient == ROTATION_180) {
        env->cap_col_2 -= 1;
    }

    int idx = env->cap_row_2*env->n_cols + env->cap_col_2;
    bool out_of_bounds = env->cap_row_2 < 0 || env->cap_row_2 >= env->n_rows
                      || env->cap_col_2 < 0 || env->cap_col_2 >= env->n_cols;
    if (env->grid[idx] != 0 || out_of_bounds) {
        env->cap_orient = old_orient;
        env->cap_row_2 = old_cap_row_2;
        env->cap_col_2 = old_cap_col_2;
        return;
    }

    env->score += SCORE_ROTATE;
    env->rewards[0] += REWARD_ROTATE;
}

void move_cap(DrMario* env) {
    env->tick_fall += 1;
    if (env->tick_fall >= env->ticks_per_fall) {
        env->tick_fall = 0;
        if (!env->cap_colliding_down) {
            env->cap_row_1 += 1;
            env->cap_row_2 += 1;
        }
        return;
    }

    if (env->cap_row_1 < 0 || env->cap_row_2 < 0) {
        return;
    }
    
    if (env->actions[0] == ACTION_LEFT && !env->cap_colliding_left) {
        env->cap_col_1 -= 1;
        env->cap_col_2 -= 1;
    } else if (env->actions[0] == ACTION_RIGHT && !env->cap_colliding_right) {
        env->cap_col_1 += 1;
        env->cap_col_2 += 1;
    } else if (env->actions[0] == ACTION_DOWN && !env->cap_colliding_down) {
        env->cap_row_1 += 1;
        env->cap_row_2 += 1;

        env->atn_count_soft_drop += 1;
        env->score += SCORE_SOFT_DROP;
        env->rewards[0] += REWARD_SOFT_DROP;
    } else if (env->actions[0] == ACTION_DROP) {
        if (!env->cap_colliding_down) {
            env->atn_count_hard_drop += 1;
            env->score += SCORE_HARD_DROP;
            env->rewards[0] += REWARD_HARD_DROP;
            do {
                env->cap_row_1 += 1;
                env->cap_row_2 += 1;
                get_collisions(env);
            } while (!env->cap_colliding_down);
        }
    }
}

void clear_lines(DrMario* env) {
    int n = 1000;
    int i = 0;

    env->lines_cleared_step = 0;
    env->viruses_cleared_step = 0;

    while (i < n) {
        bool *to_clear = (bool*)calloc(env->n_rows*env->n_cols, sizeof(bool));
        if (!to_clear) {
            break;
        }

        for (int r = 0; r < env->n_rows; r++) {
            for (int c = 0; c < env->n_cols; c++) {
                int cell = env->grid[r*env->n_cols + c];
                if (cell == 0) {
                    continue;
                }
                int color = abs(cell);
                int c_end = c + 1;
                while (c_end < env->n_cols && abs(env->grid[r*env->n_cols + c_end]) == color) {
                    c_end += 1;
                }
                if (c_end - c >= 4) {
                    env->lines_cleared_step++;
                    for (int k = c; k < c_end; k++) {
                        to_clear[r*env->n_cols + k] = true;
                    }
                }
                c = c_end-1;
            }
        }

        for (int c = 0; c < env->n_cols; c++) {
            for (int r = 0; r < env->n_rows; r++) {
                int cell = env->grid[r*env->n_cols + c];
                if (cell == 0) {
                    continue;
                }
                int color = abs(cell);
                int r_end = r + 1;
                while (r_end < env->n_rows && abs(env->grid[r_end*env->n_cols + c]) == color) {
                    r_end += 1;
                }
                if (r_end - r >= 4) {
                    env->lines_cleared_step++;
                    for (int k = r; k < r_end; k++) {
                        to_clear[k*env->n_cols + c] = true;
                    }
                }
                r = r_end-1;
            }
        }

        bool any_cleared = false;
        for (int k = 0; k < env->n_rows*env->n_cols; k++) {
            if (to_clear[k]) { 
                any_cleared = true;
                break; 
            }
        }

        if (!any_cleared) {
            free(to_clear);
            break;
        }

        for (int k = 0; k < env->n_rows*env->n_cols; k++) {
            if (to_clear[k]) {
                if (env->grid[k] < 0) {
                    env->viruses_remaining--;
                    env->viruses_cleared++;
                    env->viruses_cleared_step++;
                }
                env->grid[k] = 0;
            }
        }

        int m = 1000;
        int j = 0;
        while (j < m) {
            bool falling = false;
            for (int r = env->n_rows-2; r >= 0; r--) {
                for (int c = 0; c < env->n_cols; c++) {
                    int cell = env->grid[r*env->n_cols + c];
                    if (cell <= 0 || env->grid[(r+1)*env->n_cols + c] != 0) {
                        continue;
                    }
                    
                    bool neighbor_cleared = (r < env->n_rows-2 && to_clear[(r+1)*env->n_cols + c])
                                         || (r > 0 && to_clear[(r-1)*env->n_cols + c])
                                         || (c < env->n_cols-1 && to_clear[r*env->n_cols + c+1])
                                         || (c > 0 && to_clear[r*env->n_cols + c-1]);
                    if (neighbor_cleared) {
                        to_clear[r*env->n_cols + c] = true;
                    }

                    if (to_clear[r*env->n_cols + c]) {
                        env->grid[(r+1)*env->n_cols + c] = cell;
                        env->grid[r*env->n_cols + c] = 0;
                        falling = true;   
                    }
                }
            }
            if (!falling) {
                break;
            }
            j++;
        }

        free(to_clear);
        i++;
    }   
}

void spawn_new_cap(DrMario* env) {
    if (!env->cap_colliding_down) {
        return;
    }

    env->grid[env->cap_row_1*env->n_cols + env->cap_col_1] = env->cap_color_a;
    env->grid[env->cap_row_2*env->n_cols + env->cap_col_2] = env->cap_color_b;

    int row = env->cap_row_1 > env->cap_row_2 ? env->cap_row_1 : env->cap_row_2;
    env->rewards[0] += row*REWARD_HEIGHT;

    get_color_collisions(env);

    int color_collisions = 0;
    if (env->cap_colliding_color_hor_1 >= 2) {
        color_collisions += env->cap_colliding_color_hor_1;
    }
    if (env->cap_colliding_color_hor_2 >= 2) {
        color_collisions += env->cap_colliding_color_hor_2;
    }
    if (env->cap_colliding_color_ver_1 >= 2) {
        color_collisions += env->cap_colliding_color_ver_1;
    }
    if (env->cap_colliding_color_ver_2 >= 2) {
        color_collisions += env->cap_colliding_color_ver_2;
    }

    if (color_collisions > 0) {
        env->score += color_collisions*SCORE_PLACE_NEXT_TO_SAME_COLOR;
        env->rewards[0] += color_collisions*REWARD_PLACE_NEXT_TO_SAME_COLOR;
    }

    clear_lines(env);
    if (env->viruses_cleared_step > 0) {
        env->rewards[0] += env->viruses_cleared_step*REWARD_KILL_VIRUS;
        env->score += env->viruses_cleared_step*SCORE_KILL_VIRUS;
    }

    if (env->lines_cleared_step > 0) {
        env->rewards[0] += env->lines_cleared_step*REWARD_CLEAR_LINE;
        env->score += env->lines_cleared_step*SCORE_CLEAR_LINE;
    }

    if (env->lines_cleared_step == 0 && env->viruses_cleared_step == 0) {
        env->rewards[0] += REWARD_NO_LINE_CLEARS;
        env->score += SCORE_NO_LINE_CLEARS;
    }

    spawn_capsule(env);
}

void end_game_check(DrMario* env) {
    if (env->viruses_remaining <= 0) {
        float speed_bonus = 1.0f / (1.0f + env->tick*0.001f);
        env->rewards[0] += 1.0f + speed_bonus;
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
        return;
    }

    bool cap_at_top = env->cap_colliding_down && (env->cap_row_1 <= 0 || env->cap_row_2 <= 0);
    if (!cap_at_top) {
        return;
    }
    float fraction_remaining = env->viruses_remaining / (float)env->n_init_viruses;
    env->rewards[0] -= 1.0f + fraction_remaining*0.5f;
    env->terminals[0] = 1;
    add_log(env);
    c_reset(env);
}

void c_step(DrMario *env) {
    env->tick += 1;
    env->terminals[0] = 0;
    env->rewards[0] = 0;

    env->lines_cleared_step = 0;
    env->viruses_cleared_step = 0;

    get_collisions(env);

    if (!env->cap_colliding_down) {
        rotate_cap(env);
        get_collisions(env);
    }

    move_cap(env);   
    get_collisions(env);

    end_game_check(env);
    spawn_new_cap(env);
    
    env->episode_return += env->rewards[0];

    compute_observations(env);
}

void c_render(DrMario *env) {
    if (!IsWindowReady()) {
        InitWindow(SQUARE_SIZE*env->n_cols, SQUARE_SIZE*env->n_rows, "Dr Mario");
        SetTargetFPS(30);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(BLACK);

    for (int r = 0; r < env->n_rows; r++) {
        for (int c = 0; c < env->n_cols; c++) {
            int cell = env->grid[r*env->n_cols + c];
            int x = c*SQUARE_SIZE;
            int y = r*SQUARE_SIZE;
            if (cell == 0) {
                continue;
            }

            Color color;
            if (cell == 1 || cell == -1) {
                color = RED;
            } else if (cell == 2 || cell == -2) {
                color = BLUE;
            } else {
                color = YELLOW;
            }

            if (cell < 0) {
                DrawCircle(x + SQUARE_SIZE/2, y + SQUARE_SIZE/2, 
                          SQUARE_SIZE/2 - 2, color);
            } else {
                DrawRectangle(x + 2, y + 2, 
                             SQUARE_SIZE - 4, SQUARE_SIZE - 4, color);
            }
        }
    }

    int x1 = env->cap_col_1*SQUARE_SIZE;
    int y1 = env->cap_row_1*SQUARE_SIZE;
    int x2 = env->cap_col_2*SQUARE_SIZE;
    int y2 = env->cap_row_2*SQUARE_SIZE;
    Color ca = (env->cap_color_a == 1) ? RED : (env->cap_color_a == 2) ? BLUE : YELLOW;
    Color cb = (env->cap_color_b == 1) ? RED : (env->cap_color_b == 2) ? BLUE : YELLOW;

    DrawRectangle(x1 + 2, y1 + 2, SQUARE_SIZE - 4, SQUARE_SIZE - 4, ca);
    DrawRectangle(x2 + 2, y2 + 2, SQUARE_SIZE - 4, SQUARE_SIZE - 4, cb);

    DrawText(TextFormat("Viruses: %d", env->viruses_remaining), 4, 4, 14, WHITE);
    DrawText(TextFormat("Score: %d", env->score), 4, 20, 14, WHITE);

    EndDrawing();
}