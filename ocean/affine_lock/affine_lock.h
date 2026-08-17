#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

typedef float obs_t;
#include "pufferenv.h"
#include "affine_lock_visible_targets.h"

#define BITS 16
#define TIMER_INDEX (2 * BITS)
#define OBS_SIZE (TIMER_INDEX + 1)
#define NUM_ATNS 1
#define NUM_ACTIONS 8
#define MAX_SOLUTION_DEPTH 16
#define CURRICULUM_DEPTH_COUNT 6
#define STEP_REWARD (-0.01f) // TODO should this be in ini so it can be swept?
#define VISIBLE_TARGET_TABLE_PATH "ocean/affine_lock/generated/affine_lock_8action_visible_targets.bin"
#define ACT_SIZES {NUM_ACTIONS}
#define PUF_STEPS_PER_SEC 2

#define PERF_WEIGHTING_LINEAR 0
#define PERF_WEIGHTING_QUADRATIC 1

#define MY_VEC_INIT
#define MY_VEC_CLOSE

// TODO should this be in ini file? So it doesn't need a build to change it.
static const int CURRICULUM_DEPTHS[CURRICULUM_DEPTH_COUNT] = {2, 4, 5, 6, 8, 16};

typedef enum AffineLockAction {
    ACTION_SHIFT_LEFT = 0,
    ACTION_SHIFT_RIGHT = 1,
    ACTION_INVERT_RIGHT_7 = 2,
    ACTION_SWAP_ADJACENT_BITS = 3,
    ACTION_SWAP_ADJACENT_PAIRS = 4,
    ACTION_SWAP_NIBBLES_EACH_BYTE = 5,
    ACTION_REVERSE_EACH_NIBBLE = 6,
    ACTION_REVERSE_EACH_BYTE = 7,
} AffineLockAction;

struct Log {
    float perf;
    float score;
    float solve_rate;
    float max_depth_solve;
    float episode_return;
    float episode_length;
    float solve_steps;
    float timeout_rate;
    float solve_efficiency;
    float target_distance;
    float solved_target_distance;
    float d6_rate;
    float d6_solve_rate;
    float d8_rate;
    float d8_solve_rate;
    float d16_rate;
    float d16_solve_rate;
    float n;
};

typedef struct AffineLockShared {
    int start_depth;
    int max_depth;
    int step_grace;
    int perf_weighting;
    uint32_t mask;
    uint32_t* next;
    VisibleTargetTable visible_target_table;
    float observation_bit_patterns[256][8];
} AffineLockShared;

struct Env {
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    int num_agents;
    unsigned int rng;
    uint32_t state;
    uint32_t target;
    int step_count;
    int max_steps;
    int scramble_depth;
    int curriculum_depth; // TODO Consider only demoting by one level on a loss, perhaps both options in the ini
    int solution_length;
    int solution_actions[MAX_SOLUTION_DEPTH];
    int target_distance;
    float episode_return;
    int owns_shared;
    AffineLockShared* shared;
};
typedef Env AffineLock;

static void init_shared(AffineLockShared* shared, int start_depth, int max_depth,
        int step_grace, int perf_weighting) {
    shared->start_depth = start_depth;
    shared->max_depth = max_depth;
    shared->step_grace = step_grace;
    shared->perf_weighting = perf_weighting;
    shared->mask = (1u << BITS) - 1u;
    for (int value = 0; value < 256; value++) {
        for (int bit = 0; bit < 8; bit++) {
            shared->observation_bit_patterns[value][bit] = (value & (1 << bit)) ? 1.0f : -1.0f;
        }
    }

    uint32_t num_states = 1u << BITS;
    shared->next = (uint32_t*)calloc(
        num_states * NUM_ACTIONS, sizeof(uint32_t));

    for (uint32_t state = 0; state < num_states; state++) {
        for (int action = 0; action < NUM_ACTIONS; action++) {
            uint32_t next = state;
            switch (action) {
                case ACTION_SHIFT_LEFT:
                    next = (state >> 1) | ((state & 1u) << 15);
                    break;
                case ACTION_SHIFT_RIGHT:
                    next = ((state << 1) & 0xffffu) | ((state >> 15) & 1u);
                    break;
                case ACTION_INVERT_RIGHT_7:
                    next = state ^ 0xfe00u;
                    break;
                case ACTION_SWAP_ADJACENT_BITS:
                    next = ((state & 0x5555u) << 1) | ((state & 0xaaaau) >> 1);
                    break;
                case ACTION_SWAP_ADJACENT_PAIRS:
                    next = ((state & 0x3333u) << 2) | ((state & 0xccccu) >> 2);
                    break;
                case ACTION_SWAP_NIBBLES_EACH_BYTE:
                    next = ((state & 0x0f0fu) << 4) | ((state & 0xf0f0u) >> 4);
                    break;
                case ACTION_REVERSE_EACH_NIBBLE:
                    next = ((state & 0x5555u) << 1) | ((state & 0xaaaau) >> 1);
                    next = ((next & 0x3333u) << 2) | ((next & 0xccccu) >> 2);
                    break;
                case ACTION_REVERSE_EACH_BYTE:
                    next = ((state & 0x5555u) << 1) | ((state & 0xaaaau) >> 1);
                    next = ((next & 0x3333u) << 2) | ((next & 0xccccu) >> 2);
                    next = ((next & 0x0f0fu) << 4) | ((next & 0xf0f0u) >> 4);
                    break;
            }
            shared->next[state * NUM_ACTIONS + action] =
                next & shared->mask;
        }
    }

    assert(visible_targets_load(VISIBLE_TARGET_TABLE_PATH,
        VISIBLE_TARGET_8ACTION_V1_HASH, &shared->visible_target_table) == 0 &&
        "failed to load visible target table -- see 'Regenerating the Target "
        "Table' in ocean/affine_lock/README.md");
}

static AffineLockShared* create_shared(int start_depth, int max_depth,
        int step_grace, int perf_weighting) {
    AffineLockShared* shared = (AffineLockShared*)calloc(1, sizeof(AffineLockShared));
    init_shared(shared, start_depth, max_depth, step_grace, perf_weighting);
    return shared;
}

static void init_env(AffineLock* env, AffineLockShared* shared, unsigned int seed) {
    env->shared = shared;
    env->rng = seed;
    env->num_agents = 1;
    env->curriculum_depth = shared->start_depth;
}

void puf_init(Env* env, Dict* kwargs) {
    int start_depth = dict_get(kwargs, "start_depth");
    int max_depth = dict_get(kwargs, "max_depth");
    int step_grace = dict_get(kwargs, "step_grace");
    int perf_weighting = dict_get(kwargs, "perf_weighting");
    unsigned int seed = (unsigned int)dict_get(kwargs, "seed");
    AffineLockShared* shared =
        create_shared(start_depth, max_depth, step_grace, perf_weighting);
    init_env(env, shared, rand_r(&seed));
    env->owns_shared = 1;
}

static void free_shared(AffineLockShared* shared) {
    free(shared->next);
    visible_targets_free(&shared->visible_target_table);
}

void puf_close(AffineLock* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
    if (env->owns_shared) {
        free_shared(env->shared);
        free(env->shared);
    }
}

static void add_log(AffineLock* env, int solved) {
    AffineLockShared* shared = env->shared;
    int log_depth = env->target_distance;
    int at_max_depth = log_depth == shared->max_depth;
    float ratio = log_depth / (float)shared->max_depth;
    float solve_credit = 0;
    if (solved) {
        solve_credit = shared->perf_weighting == PERF_WEIGHTING_QUADRATIC ?
            ratio * ratio : ratio;
    }
    env->log.perf += solve_credit;
    env->log.score += solve_credit;
    env->log.solve_rate += solved;
    env->log.max_depth_solve += solved && at_max_depth;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->step_count;
    env->log.solve_steps += solved ? env->step_count : 0;
    env->log.timeout_rate += !solved;
    env->log.solve_efficiency += solved ? env->step_count / (float)log_depth : 0;
    env->log.target_distance += env->target_distance;
    env->log.solved_target_distance += solved ? env->target_distance : 0;
    env->log.d6_rate += log_depth == 6;
    env->log.d6_solve_rate += solved && log_depth == 6;
    env->log.d8_rate += log_depth == 8;
    env->log.d8_solve_rate += solved && log_depth == 8;
    env->log.d16_rate += log_depth == 16;
    env->log.d16_solve_rate += solved && log_depth == 16;
    env->log.n += 1;
}

// Not rand_r(): glibc's LCG has statistically weak low-order bits, and this
// env repeatedly samples individual state bits and small action ranges
// directly from those bits, where the weakness would show up as bias.
static uint32_t random_mixed_u32(AffineLock* env) {
    env->rng = env->rng * 1664525u + 1013904223u;
    uint32_t x = env->rng;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static int random_bounded(AffineLock* env, int bound) {
    uint32_t ubound = bound;
    uint32_t limit = UINT32_MAX - UINT32_MAX % ubound;
    uint32_t value = random_mixed_u32(env);
    while (value >= limit) {
        value = random_mixed_u32(env);
    }
    return value % ubound;
}

static const VisibleTargetDepth* visible_target_depth(
        const AffineLockShared* shared,
        uint32_t requested_depth) {
    const VisibleTargetTable* table = &shared->visible_target_table;
    for (uint32_t i = 0; i < table->depth_count; i++) {
        if (table->depths[i].depth == requested_depth) {
            return &table->depths[i];
        }
    }
    return NULL;
}

static void reset_state(AffineLock* env) {
    AffineLockShared* shared = env->shared;
    env->scramble_depth = env->curriculum_depth;
    env->step_count = 0;
    env->episode_return = 0;
    const VisibleTargetDepth* depth = visible_target_depth(shared, env->scramble_depth);
    int choice = random_bounded(env, depth->stored_count);
    const VisibleTargetRecord* record = &shared->visible_target_table.records[depth->first_record + choice];
    env->state = record->start;
    env->target = record->target;
    env->target_distance = record->depth;
    env->solution_length = record->solution_length;
    for (int i = 0; i < MAX_SOLUTION_DEPTH; i++) {
        env->solution_actions[i] = -1;
    }
    for (int i = 0; i < env->solution_length; i++) {
        env->solution_actions[i] = (record->packed_actions >> (3 * i)) & 7;
    }
    env->max_steps = env->target_distance + shared->step_grace;
}

static void compute_observations(AffineLock* env) {
    float (*patterns)[8] = env->shared->observation_bit_patterns;
    uint32_t state = env->state;
    uint32_t target = env->target;
    float* obs = env->agents[0].observations;
    for (int i = 0; i < 8; i++) {
        obs[i] = patterns[state & 0xffu][i];
        obs[8 + i] = patterns[(state >> 8) & 0xffu][i];
        obs[16 + i] = patterns[target & 0xffu][i];
        obs[24 + i] = patterns[(target >> 8) & 0xffu][i];
    }
    obs[TIMER_INDEX] = env->step_count / (float)env->max_steps;
}

void puf_reset(AffineLock* env) {
    env->agents[0].rewards[0] = 0;
    env->agents[0].terminals[0] = 0;
    reset_state(env);
    compute_observations(env);
}

static int next_curriculum_depth( const AffineLockShared* shared, int current_depth) {
    for (int i = 0; i < CURRICULUM_DEPTH_COUNT; i++) {
        int depth = CURRICULUM_DEPTHS[i];
        if (depth > current_depth) {
            return depth < shared->max_depth ? depth : shared->max_depth;
        }
    }
    return shared->max_depth;
}

static int human_controls(AffineLock *env) {
    if (!IsWindowReady() || !IsKeyDown(KEY_LEFT_SHIFT)) {
        return 0;
    }
    static const int keys[NUM_ACTIONS] = {
        KEY_ONE, KEY_TWO, KEY_THREE, KEY_FOUR,
        KEY_FIVE, KEY_SIX, KEY_SEVEN, KEY_EIGHT,
    };
    for (int i = 0; i < NUM_ACTIONS; i++) {
        if (IsKeyPressed(keys[i])) {
            env->agents[0].actions[0] = (float)i;
            return 1;
        }
    }
    return -1;
}

void puf_step(AffineLock* env) {
    if (human_controls(env) < 0) {
        return;
    }
    AffineLockShared* shared = env->shared;
    float reward = STEP_REWARD;
    int terminal = 0;
    int solved = 0;

    env->agents[0].terminals[0] = 0;
    env->step_count += 1;
    float raw = env->agents[0].actions[0];
    int invalid = !isfinite(raw) || raw < 0 || raw > NUM_ACTIONS - 1;
    if (invalid) {
        reward = -1;
        terminal = 1;
    } else {
        int action = (int)raw;
        env->state = shared->next[env->state * NUM_ACTIONS + action];
        if (env->state == env->target) {
            reward = 1;
            terminal = 1;
            solved = 1;
        } else if (env->step_count >= env->max_steps) {
            reward = -1;
            terminal = 1;
        }
    }
    env->agents[0].rewards[0] = reward;
    env->episode_return += reward;
    if (terminal) {
        env->agents[0].terminals[0] = 1;
        add_log(env, solved);
        env->curriculum_depth = solved ?
            next_curriculum_depth(shared, env->scramble_depth) :
            shared->start_depth;
        reset_state(env);
    }
    compute_observations(env);
}

void puf_log(Log* log, Dict* out) {
    float nsolve = log->solve_rate;
    float solved_min_win_moves = nsolve ? log->solved_target_distance / nsolve : 0;
    float conditional_solve_steps = nsolve ? log->solve_steps / nsolve : 0;
    float conditional_solve_efficiency = nsolve ? log->solve_efficiency / nsolve : 0;

    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "solve_rate", log->solve_rate);
    dict_set(out, "max_depth_solve", log->max_depth_solve);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "timeout_rate", log->timeout_rate);
    dict_set(out, "min_win_moves", log->target_distance);
    dict_set(out, "solved_min_win_moves", solved_min_win_moves);
    dict_set(out, "conditional_solve_steps", conditional_solve_steps);
    dict_set(out, "conditional_solve_efficiency", conditional_solve_efficiency);
    dict_set(out, "d6_solve_rate", log->d6_rate ? log->d6_solve_rate / log->d6_rate : 0);
    dict_set(out, "d8_solve_rate", log->d8_rate ? log->d8_solve_rate / log->d8_rate : 0);
    dict_set(out, "d16_solve_rate", log->d16_rate ? log->d16_solve_rate / log->d16_rate : 0);
    dict_set(out, "n", log->n);
}

Env* my_vec_init(int* num_envs_out, int* buffer_env_starts, int* buffer_env_counts,
        Dict* vec_kwargs, Dict* env_kwargs) {
    int total_agents = dict_get(vec_kwargs, "total_agents");
    int num_buffers = dict_get(vec_kwargs, "num_buffers");
    int agents_per_buffer = total_agents / num_buffers;
    unsigned int running_seed = (unsigned int)dict_get(env_kwargs, "seed");
    int start_depth = dict_get(env_kwargs, "start_depth");
    int max_depth = dict_get(env_kwargs, "max_depth");
    int step_grace = dict_get(env_kwargs, "step_grace");
    int perf_weighting = dict_get(env_kwargs, "perf_weighting");

    AffineLockShared* shared =
        create_shared(start_depth, max_depth, step_grace, perf_weighting);
    Env* envs = (Env*)calloc(total_agents, sizeof(Env));

    int buf = 0;
    int buf_agents = 0;
    buffer_env_starts[0] = 0;
    buffer_env_counts[0] = 0;
    for (int i = 0; i < total_agents; i++) {
        Env* env = &envs[i];
        init_env(env, shared, rand_r(&running_seed));
        buf_agents += env->num_agents;
        buffer_env_counts[buf]++;
        if (buf_agents >= agents_per_buffer && buf < num_buffers - 1) {
            buf++;
            buffer_env_starts[buf] = i + 1;
            buffer_env_counts[buf] = 0;
            buf_agents = 0;
        }
    }
    *num_envs_out = total_agents;
    return envs;
}

void my_vec_close(Env* envs) {
    free_shared(envs[0].shared);
    free(envs[0].shared);
}

void puf_render(AffineLock* env) {
    if (IsWindowReady() && (WindowShouldClose() || IsKeyPressed(KEY_ESCAPE))) {
        puf_close(env);
        exit(0);
    }
    human_controls(env);
    static int window_initialized = 0;
    if (!window_initialized) {
        InitWindow(780, 360, "PufferLib AffineLock");
        SetTargetFPS(30);
        window_initialized = 1;
    }

    uint32_t rel = (env->state ^ env->target) & env->shared->mask;
    float display_reward = env->agents[0].rewards[0];
    int display_terminal = env->agents[0].terminals[0] != 0;
    int display_solved = display_terminal && display_reward > 0;
    const char* status = "running";
    Color status_color = (Color){190, 198, 206, 255};
    if (display_terminal) {
        status = display_solved ? "solved" : "failed";
        status_color = display_solved ?
            (Color){80, 210, 140, 255} : (Color){238, 88, 88, 255};
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    DrawText("Affine Lock", 30, 24, 28, RAYWHITE);
    DrawText(TextFormat("depth %d/%d  step %d/%d  last reward %.2f",
        env->scramble_depth, env->shared->max_depth,
        env->step_count, env->max_steps, display_reward),
        30, 62, 20, (Color){180, 190, 200, 255});
    DrawText(TextFormat("status %s  mismatches 0x%04x",
        status, rel), 30, 90, 20, status_color);

    const char* row_label[2] = {"current", "target"};
    uint32_t row_value[2] = {env->state, env->target};
    int row_y[2] = {138, 220};
    for (int row = 0; row < 2; row++) {
        DrawText(row_label[row], 30, row_y[row] + 9, 20, RAYWHITE);
        for (int bit = 0; bit < BITS; bit++) {
            int x = 145 + bit * 34;
            int on = (row_value[row] >> bit) & 1u;
            int mismatch = ((env->state ^ env->target) >> bit) & 1u;
            Color fill = on ? (Color){80, 210, 140, 255} : (Color){38, 48, 58, 255};
            Color border = mismatch ? (Color){238, 88, 88, 255} : (Color){182, 196, 205, 255};
            DrawRectangle(x, row_y[row], 24, 34, fill);
            DrawRectangleLinesEx(
                (Rectangle){x, row_y[row], 24, 34},
                mismatch ? 3 : 1, border);
            DrawText(TextFormat("%d", bit), x + 5, row_y[row] + 40, 10,
                (Color){128, 140, 150, 255});
        }
    }

    DrawText("1 shiftL  2 shiftR  3 inv7  4 bit-swap  5 pair-swap",
        30, 300, 16, (Color){160, 170, 178, 255});
    DrawText("6 nib-swap  7 rev-nib  8 rev-byte  R reset",
        30, 322, 16, (Color){160, 170, 178, 255});
    EndDrawing();
    puf_web_vsync();
}
